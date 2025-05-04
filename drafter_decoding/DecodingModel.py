import time
from typing import Tuple

import torch.nn as nn
import torch

from .CacherManager import CacheManager
from .Config import Config
from .KVCacheModel import KVCacheModel
from .Tree import Tree
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from drafter_decoding.kv_cache import initialize_past_key_values
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.distributed as dist

from .util import seed_everything, InferenceData


def color_print(content: str, color_number: int = 4):
    """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
    # if self.accelerator.is_main_process:
    print(f"\033[9{color_number}m{content}\033[0m")


class DecodingModel(nn.Module):
    def __init__(self,
                 model,
                 model_name_or_path,
                 device: torch.device,
                 parser_args = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if parser_args is not None:
            seed_everything(parser_args.seed)
            self.parser_args = parser_args
            self.seed = parser_args.seed

            # 从 args 中获取分布式参数
            self.rank = parser_args.rank  # 当前进程的全局排名
            self.world_size = parser_args.world_size  # 总进程数
            self.local_rank = parser_args.local_rank  # 本地设备上的排名（如单机多卡时为 GPU ID）

            self.is_drafter: bool = False
            self.is_target_model = False
            self.cuda_device_count: int = torch.cuda.device_count()
            dist.init_process_group(
                backend='nccl',  # 如果是纯 CPU 用 'gloo'，GPU 建议 'nccl'
                init_method='tcp://127.0.0.1:12345',  # 或者从 args 传入
                rank=self.rank,
                world_size=self.world_size
            )
            self.eval_mode = parser_args.eval_mode
            if parser_args.eval_mode == "two_model":
                self._init_two_model_config()
            elif parser_args.eval_mode == "three_model":
                pass
            elif parser_args.eval_mode == "single_model":
                self._init_single_model_config()
            # load_model
            self._load_model()
            # self.inference_data = InferenceData(self.rank)
            self.hidden_size = self.model.lm_head.weight.shape[-1]
            self.vocab_size = self.model.lm_head.weight.shape[0]
            self.tokenizer = AutoTokenizer.from_pretrained(self.parser_args.target_model_dir)
            self.is_llama: bool = False
            self.is_qwen: bool = False
            if 'llama' in self.parser_args.target_model_dir.lower(): # default is llama
                self.is_llama = True
            if 'qwen' in self.parser_args.target_model_dir.lower():
                self.is_qwen = True
            self.model_name = parser_args.model_name
            self.max_tokens = parser_args.max_tokens
            self.temperature = parser_args.temperature
            self.top_k = parser_args.top_k
            self.top_p = parser_args.top_p
            self.branch_prediction_num = parser_args.branch_prediction_num
            self.drafters_num = len(parser_args.draft_models_dir)
            self.target_model_rank_num = self.drafters_num
            self.dataset_name = ""
            self.verified_len: int = 0
            self.inference_data = InferenceData(self.rank)
            dist.barrier()
            return
        self.model = model
        self.hidden_size = model.lm_head.weight.shape[-1]
        self.vocab_size = model.lm_head.weight.shape[0]
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.verified_len: int = 0

    def _init_two_model_config(self) -> None:
        color_print(f"初始化", self.rank)
        if self.local_rank == 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_drafter = True
        elif self.local_rank == 1 and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_target_model = True

    def _init_single_model_config(self) -> None:
        color_print(f" single model running...")
        self.is_target_model = True
        self.device = torch.device(f"cuda:0")

    def _load_model(self):
        torch.set_grad_enabled(False)
        if self.eval_mode == "two_model":
            print(f"进程 {self.local_rank} 的本地设备: {self.device}")
            # 所有模型仅用于推理，禁用梯度
            if self.is_drafter:
                color_print(f"{self.device} Loading models:{self.parser_args.draft_models_dir[self.local_rank]}\n",
                                 self.rank)
                # Draft 模型：严格绑定到当前设备
                # caution drafter should use KVLlamaForCausalLM as the model
                self.model = KVLlamaForCausalLM.from_pretrained(
                    self.parser_args.draft_models_dir[self.local_rank],
                    device_map={"": self.device},  # 显式指定GPU索引
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
            if self.is_target_model:
                config = AutoConfig.from_pretrained(self.parser_args.target_model_dir)
                device_map = {"": self.device}
                if self.cuda_device_count == 3:
                    num_layers = config.num_hidden_layers
                    split_point = num_layers // 2 + 1
                    device_map = {
                        "model.embed_tokens": "cuda:1",
                        **{f"model.layers.{i}": "cuda:1" for i in range(0, split_point)},
                        **{f"model.layers.{i}": "cuda:2" for i in range(split_point, num_layers)},
                        "model.norm": "cuda:2",
                        "model.rotary_emb": "cuda:2",
                        "lm_head": "cuda:2",
                    }
                print(device_map)
                color_print(f"Loading models:{self.parser_args.target_model_dir}\n", self.rank)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.parser_args.target_model_dir,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).eval()
                # print(self.model)
        if self.eval_mode == "single_model":
            color_print(f"loading model from {self.parser_args.target_model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.parser_args.target_model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).eval()

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            main_device: str,
            draft_model_path=None,
            target_model_path=None,
            **kwargs,
    ):
        model_name = 'llama'
        # drafter using the customized model
        if draft_model_path is not None:
            if 'llama' in draft_model_path.lower():
                print(f" load llama model")
                model = KVLlamaForCausalLM.from_pretrained(
                    draft_model_path, **kwargs
                ).eval()
            if 'qwen' in draft_model_path.lower():
                model_name = 'qwen'
                print(f" load qwen model")
                model = KVQwen2ForCausalLM.from_pretrained(
                    draft_model_path, **kwargs
                ).eval()

        # target model using the class from transformers lib on huggingface
        elif target_model_path is not None:
            model = AutoModelForCausalLM.from_pretrained(target_model_path,
                                                         **kwargs).eval()
        # cls() 调用 MyClass 的 __init__ 方法创建了一个新实例
        ret = cls(model, target_model_path, device=main_device)
        ret.model_name = model_name
        return ret

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        # if self.accelerator.is_main_process:
        print(f"\033[9{color_number}m{content}\033[0m")

    def process_tree_mask(self, tree_attention_mask, init_len):
        # todo 考虑封装到 Tree 中
        attention_mask = torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask = torch.cat([attention_mask, tree_mask], dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def autoregressive_decoding(self,
                              input_ids: torch.Tensor) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Store the original input to preserve it during generation
        original_input_length = input_ids.shape[1]

        # Set generation parameters
        temperature = 1.0  # Control randomness (lower = more deterministic)

        # Generation loop
        output_ids = input_ids.clone()

        for _ in range(self.max_tokens):
            # Get only the most recent context to avoid excessive memory usage
            # Optional, especially useful for longer generations
            if output_ids.shape[1] > 1024:  # Example context window size
                context_ids = output_ids[:, -1024:]
            else:
                context_ids = output_ids

            with torch.no_grad():  # No need to track gradients during inference
                outputs = self.model(context_ids)

            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Optional: Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Get the most likely next token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Append the new token to the sequence
            output_ids = torch.cat([output_ids, next_token.transpose(0, 1)], dim=1)

            # Print the current output (optional - can be removed for performance)
            current_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Generated so far: {current_text}")

            # Stop if we generate an EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Get the final generated text
        generated_text = self.tokenizer.decode(output_ids[0][original_input_length:], skip_special_tokens=True)
        print(f"\nFinal generated text:\n{generated_text}")
        sum_tokens = output_ids[0][original_input_length:].shape[0]
        return output_ids

    # draft 是一个不断进行树状起草的函数，采用预先分配的kv cache 进行起草工作
    @torch.no_grad()
    def draft_single_card_test(self,
                               input_ids: torch.Tensor,
                               nodes_per_layer: int = 20,
                               max_depth: int = 50,
                               ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data_list
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        tree_attention_mask = None
        print(self.tokenizer.decode(input_ids[0]))

        # prefill 阶段
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            tree_attention_mask=tree_attention_mask,
            past_key_values=past_key_values,
            position_ids=None,
        )
        # init_len用于制作掩码
        init_len = input_ids.shape[1]
        self.verified_len = init_len
        verified_len_posi = init_len - 1

        # 设置树所在的设备
        tree = Tree(verified_len_posi, outputs[0].device, nodes_per_layer=nodes_per_layer, max_depth=max_depth)
        # cache_manager = CacheManager(world_size=self.world_size,rank=self.rank,device=tree.device,tree=tree,is_drafter=True)
        # decode 阶段，在 decode 阶段，只要没有被完全拒绝，每一次都要处理一层的树节点。
        # 通过 Tree 类进行管理
        debug_queue = []

        input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
            torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
        tree_attention_mask = self.process_tree_mask(tree_attention_mask, init_len)
        # drafter decoding
        j = 0
        start_time = time.perf_counter()
        while True:
            # if self.tokenizer.eos_token_id == input_ids:
            #     break
            # input_ids = input_ids.unsqueeze(0).unsqueeze(0)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,
                tree_attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            # 测试回滚效果
            if tree.size >= 20:
                tokens, index = tree.pick_path_for_test()
                tensor1 = torch.tensor(len(index), device='cuda').unsqueeze(0)
                tensor2 = torch.tensor(799, device='cuda').unsqueeze(0)
                tensor3 = torch.tensor(index, device='cuda')
                buffer = torch.cat([tensor1, tensor2, tensor3], dim=-1)
                outputs = self._verified_update(tree,
                                                buffer,
                                                outputs)

                debug_queue.extend(tokens)

            # outputs[0]： 表示的是 logits
            # 第一次推理节点的维度为 size(1,23,32000)
            # size(batch_size, seq_len, vocab_size)
            # 可以根据 seq_len 来判断当前处理情况是一层树，还是只是正常推理
            input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
                torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
            tree_attention_mask = self.process_tree_mask(tree_attention_mask, self.verified_len)

            # candidates_id.append(input_ids[0][1].item())
            # print(self.tokenizer.decode(input_ids[0]))
            print(f"the best candidates are \n{self.tokenizer.decode(debug_queue)}")
            print(f"the token_ids are {debug_queue}")
            print(f"the total len is {len(debug_queue)}")
            print(f"{j}..{(time.perf_counter() - start_time)}............")
            j += 1

    # draft 是一个不断进行树状起草的函数，采用预先分配的kv cache 进行起草工作
    @torch.no_grad()
    def draft_single_card_test_qwen(self,
                               input_ids: torch.Tensor,
                               nodes_per_layer: int = 20,
                               max_depth: int = 50,
                               ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data_list
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        tree_attention_mask = None
        print(self.tokenizer.decode(input_ids[0]))

        # prefill 阶段
        outputs = self.model(
            input_ids=input_ids,
            # attention_mask=tree_attention_mask,
            past_key_values=past_key_values,
            cache_position=None,
            use_cache=False,
        )
        # init_len用于制作掩码
        init_len = input_ids.shape[1]
        self.verified_len = init_len
        verified_len_posi = init_len - 1

        # 设置树所在的设备
        tree = Tree(verified_len_posi, outputs[0].device, nodes_per_layer=nodes_per_layer, max_depth=max_depth)
        # cache_manager = CacheManager(world_size=self.world_size,rank=self.rank,device=tree.device,tree=tree,is_drafter=True)
        # decode 阶段，在 decode 阶段，只要没有被完全拒绝，每一次都要处理一层的树节点。
        # 通过 Tree 类进行管理
        debug_queue = []

        input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
            torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
        # position_ids = position_ids.unsqueeze(0)
        tree_attention_mask = self.process_tree_mask(tree_attention_mask, init_len)
        # drafter decoding
        j = 0
        start_time = time.perf_counter()
        while True:
            # if self.tokenizer.eos_token_id == input_ids:
            #     break
            # input_ids = input_ids.unsqueeze(0).unsqueeze(0)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                cache_position=position_ids,
            )
            # 测试回滚效果
            if tree.size >= 2:
                tokens, index = tree.pick_path_for_test()
                tensor1 = torch.tensor(len(index), device='cuda').unsqueeze(0)
                tensor2 = torch.tensor(0, device='cuda').unsqueeze(0)
                tensor3 = torch.tensor(index, device='cuda')
                buffer = torch.cat([tensor1, tensor2, tensor3], dim=-1)
                outputs = self._verified_update(tree,
                                                buffer,
                                                outputs)

                debug_queue.extend(tokens)

            # outputs[0]： 表示的是 logits
            # 第一次推理节点的维度为 size(1,23,32000)
            # size(batch_size, seq_len, vocab_size)
            # 可以根据 seq_len 来判断当前处理情况是一层树，还是只是正常推理
            input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
                torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
            # position_ids = position_ids.unsqueeze(0)
            tree_attention_mask = self.process_tree_mask(tree_attention_mask, self.verified_len)

            # candidates_id.append(input_ids[0][1].item())
            # print(self.tokenizer.decode(input_ids[0]))
            print(f"the best candidates are \n{self.tokenizer.decode(debug_queue)}")
            print(f"the token_ids are {debug_queue}")
            print(f"the total len is {len(debug_queue)}")
            print(f"{j}..{(time.perf_counter() - start_time)}............")
            j += 1

    # draft 是一个不断进行树状起草的函数，采用预先分配的kv cache 进行起草工作
    @torch.no_grad()
    def draft(self,
              input_ids: torch.Tensor,
              nodes_per_layer: int = 20,
              max_depth: int = 50,
              ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data_list
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        tree_attention_mask = None
        print(self.tokenizer.decode(input_ids[0]))

        # prefill 阶段
        if self.is_llama:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,
                tree_attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                position_ids=None,
            )
        if self.is_qwen:
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
            )
        # init_len用于制作掩码
        init_len = input_ids.shape[1]
        self.verified_len = init_len
        verified_len_posi = init_len - 1

        # 设置树所在的设备
        tree = Tree(verified_len_posi, outputs[0].device, nodes_per_layer=nodes_per_layer, max_depth=max_depth)
        cache_manager = CacheManager(world_size=self.world_size,rank=self.rank,device=tree.device,tree=tree,is_drafter=True)
        # decode 阶段，在 decode 阶段，只要没有被完全拒绝，每一次都要处理一层的树节点。
        # 通过 Tree 类进行管理
        input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
            torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
        tree_attention_mask = self.process_tree_mask(tree_attention_mask, init_len)
        # drafter decoding
        while cache_manager.is_decoding:
            # if self.tokenizer.eos_token_id == input_ids:
            #     break
            # input_ids = input_ids.unsqueeze(0).unsqueeze(0)
            if self.is_llama:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    tree_attention_mask=tree_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            if self.is_qwen:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=tree_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=position_ids,
                )


            if cache_manager.is_update:
                # we need to update the tree buffer
                with cache_manager.lock:
                    outputs = self._verified_update(tree,cache_manager.recv_buffer,outputs)
                    cache_manager.is_update = False

            # outputs[0]： 表示的是 logits
            # 第一次推理节点的维度为 size(1,23,32000)
            # size(batch_size, seq_len, vocab_size)
            # 可以根据 seq_len 来判断当前处理情况是一层树，还是只是正常推理
            input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
                torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
            tree_attention_mask = self.process_tree_mask(tree_attention_mask, self.verified_len)


    @torch.no_grad()
    def decoding_with_cache(self,
                            input_ids: torch.Tensor,
                            nodes_per_layer: int = 20,
                            max_depth: int = 50):
        '''
        decoding with tree_cache for target model
        Returns:

        '''
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        self.verified_len = input_ids.shape[1]
        # create tree ,start recv cache
        tree_buffer = Tree(input_ids.shape[1], self.device, nodes_per_layer, max_depth)

        cache_manager = CacheManager(self.world_size, self.rank, self.device, tree_buffer, is_target_model=True)

        # prefill phase merged into decoding phase
        init_len = input_ids.shape[1]
        model = KVCacheModel(self.model, init_input_len=init_len)
        # Initialize the past key and value states
        # input_ids = model.generate(input_ids)
        # decode phase
        output_token_ids = input_ids.clone()
        # 在第一次进行 decode 过程先查询 cache
        query_flag = torch.zeros(1, device=model.device,dtype=torch.int)
        dist.isend(query_flag, dst=Config.DRAFTER_RANK)
        cache_manager.recv_buffer_for_target_model()
        cache_manager.update_cache_for_target_model()
        hit_cache = True
        send_verified_index_buffer = torch.zeros(tree_buffer.buffer_capacity + 2, device=self.device, dtype=torch.int)
        start = time.perf_counter()
        max_length = 500
        for i in range(max_length):
            if hit_cache is False:
                # 被拒绝重新请求 cache
                # color_print(f"hit_cache is {hit_cache}", self.rank)
                dist.send(query_flag, dst=Config.DRAFTER_RANK)
                cache_manager.recv_buffer_for_target_model()
                cache_manager.update_cache_for_target_model()
            # 1. query cache
            best_candidates, picked_index, root_index = cache_manager.query_cache()
            # color_print(f"tree size 大小为{tree_buffer.size}\ntarget model 查询cache 结果为best_candidates, picked_index, root_index:{best_candidates, picked_index, root_index}",self.rank)
            best_candidates = torch.tensor(best_candidates, device=self.device, dtype=torch.int).unsqueeze(0)
            # 2. decode with cache
            if model._past_key_values is None:
                best_candidates = torch.cat([input_ids, best_candidates], dim=-1)
            else:
                best_candidates = torch.cat([new_sample_token.unsqueeze(0),best_candidates],dim=-1)
            token_ids = model.generate(best_candidates)

            # color_print(f"target model 生成 token_ids:{token_ids}",self.rank)
            cache_manager.recv_buffer_for_target_model()
            # [debug]
            # 1d token_ids
            # 保存输出
            output_token_ids = torch.cat([output_token_ids, token_ids.unsqueeze(0)], dim=-1)
            # color_print(f"len is {output_token_ids.shape[1]}output_token_ids is {output_token_ids}",self.rank)
            # color_print(self.tokenizer.decode(output_token_ids[0]),5)
            # tokens_ids is 1-d tensor
            # 3. check if all verified tokens in cache
            # let me clear: what is accept and reject?
            # as long as all the tokens_ids are in tree_buffer, that means accept .the other will be rejection
            # 3.1 update tree buffer
            correct_ids_index_path:list | torch.Tensor = picked_index[:token_ids.shape[0] - 1]
            new_sample_token = token_ids[-1].unsqueeze(0)
            cache_manager.update_cache_for_target_model()
            hit_cache = cache_manager.update_tree_buffer(correct_ids_index_path, new_sample_token,
                                                         root_index=root_index)
            correct_ids_index_path = torch.tensor(correct_ids_index_path,device=self.device)
            # color_print(f"correct_ids_index_path is {correct_ids_index_path}",self.rank)
            # hit_cache 如果不为 false 则其为 一个 tensor 里面表示 new_token_idx
            if hit_cache is not False:
                # 命中cache
                #   3.1 acc (could find a path in cache)
                seq_len = torch.tensor(token_ids.shape[0], device=self.device).unsqueeze(0)
                pad = torch.tensor(-1, device=self.device).unsqueeze(0)
                send_msg = torch.cat([seq_len, pad, correct_ids_index_path, hit_cache], dim=-1).to(torch.int)
            else:
                # 未命中，发送格式
                # index[0] = -1 序列被拒绝
                # index[1] 被拒绝后的token
                #   3.2 reject  (could not find a path in cache)
                seq_len = torch.tensor(correct_ids_index_path.shape[0], device=self.device).unsqueeze(0)
                # color_print(f"seq_len, new_sample_token, correct_ids_index_path is {seq_len, new_sample_token, correct_ids_index_path}",self.rank)
                send_msg = torch.cat([seq_len, new_sample_token, correct_ids_index_path], dim=-1).to(torch.int)

            # 4. isend message including index info to drafter
            # color_print(f"target model send_msg is {send_msg}",self.rank)
            send_verified_index_buffer[0:send_msg.shape[0]].copy_(send_msg)
            # dist.send(send_msg, dst=Config.DRAFTER_RANK)
            dist.send(send_verified_index_buffer, dst=Config.DRAFTER_RANK)
            # currently send a 1-d tensor todo to align the dim with the receiver
            # after the last communication , check if to end. 一定在这个位置 check，要不然会不正常结束
            if self.tokenizer.eos_token_id in token_ids.tolist():
                break
            # 5.  rollback kv_cache 不能算上 新 sample 出来的token
            model.rollback(output_token_ids.shape[1] - 1)
            color_print(f"tokens/s is {(output_token_ids.shape[1] - init_len) / (time.perf_counter() - start)}\n output_len is {output_token_ids.shape[1]}",5)
        return output_token_ids

    @torch.no_grad()
    def decoding_with_cache_profile(self,
                            input_ids: torch.Tensor,
                            nodes_per_layer: int = 20,
                            max_depth: int = 50):
        '''
        decoding with tree_cache for target model
        Returns:

        '''
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        self.verified_len = input_ids.shape[1]
        # create tree ,start recv cache
        tree_buffer = Tree(input_ids.shape[1], self.device, nodes_per_layer, max_depth)

        cache_manager = CacheManager(self.world_size, self.rank, self.device, tree_buffer, is_target_model=True)

        # prefill phase merged into decoding phase
        init_len = input_ids.shape[1]
        model = KVCacheModel(self.model, init_input_len=init_len)
        # Initialize the past key and value states
        # input_ids = model.generate(input_ids)
        # decode phase
        output_token_ids = input_ids.clone()
        # 在第一次进行 decode 过程先查询 cache
        query_flag = torch.zeros(1, device=model.device, dtype=torch.int)
        dist.isend(query_flag, dst=Config.DRAFTER_RANK)
        cache_manager.recv_buffer_for_target_model()
        cache_manager.update_cache_for_target_model()
        hit_cache = True
        send_verified_index_buffer = torch.zeros(tree_buffer.buffer_capacity + 2, device=self.device, dtype=torch.int)
        start = time.perf_counter()
        max_length = 1000
        while output_token_ids.shape[1] < max_length:
            if hit_cache is False:
                # 被拒绝重新请求 cache
                color_print(f"hit_cache is {hit_cache}", self.rank)
                # with self.inference_data.communication_timer:
                dist.send(query_flag, dst=Config.DRAFTER_RANK)
                cache_manager.recv_buffer_for_target_model()
                cache_manager.update_cache_for_target_model()
            # 1. query cache
            with self.inference_data.generate_timer:
                best_candidates, picked_index, root_index = cache_manager.query_cache()
                best_candidates = torch.tensor(best_candidates, device=self.device, dtype=torch.int).unsqueeze(0)
                # 2. decode with cache
                if model._past_key_values is None:
                    best_candidates = torch.cat([input_ids, best_candidates], dim=-1)
                else:
                    best_candidates = torch.cat([new_sample_token.unsqueeze(0), best_candidates], dim=-1)
                token_ids = model.generate(best_candidates)

            cache_manager.recv_buffer_for_target_model()
            # [debug]
            # 1d token_ids
            # 保存输出
            output_token_ids = torch.cat([output_token_ids, token_ids.unsqueeze(0)], dim=-1)
            color_print(f"target model {self.tokenizer.decode(output_token_ids[0])}",5)
            # tokens_ids is 1-d tensor
            # 3. check if all verified tokens in cache
            # let me clear: what is accept and reject?
            # as long as all the tokens_ids are in tree_buffer, that means accept .the other will be rejection
            # 3.1 update tree buffer
            correct_ids_index_path: list | torch.Tensor = picked_index[:token_ids.shape[0] - 1]
            new_sample_token = token_ids[-1].unsqueeze(0)
            with self.inference_data.communication_timer:
                cache_manager.update_cache_for_target_model()
            with self.inference_data.verification_timer:
                hit_cache = cache_manager.update_tree_buffer(correct_ids_index_path, new_sample_token,
                                                             root_index=root_index)
                correct_ids_index_path = torch.tensor(correct_ids_index_path, device=self.device)
                # hit_cache 如果不为 false 则其为 一个 tensor 里面表示 new_token_idx
                if hit_cache is not False:
                    # 命中cache
                    #   3.1 acc (could find a path in cache)
                    seq_len = torch.tensor(token_ids.shape[0], device=self.device).unsqueeze(0)
                    pad = torch.tensor(-1, device=self.device).unsqueeze(0)
                    send_msg = torch.cat([seq_len, pad, correct_ids_index_path, hit_cache], dim=-1).to(torch.int)
                    self.inference_data.add_acc(seq_len.item() - 1)
                else:
                    # 未命中，发送格式
                    # index[0] = -1 序列被拒绝
                    # index[1] 被拒绝后的token
                    #   3.2 reject  (could not find a path in cache)
                    seq_len = torch.tensor(correct_ids_index_path.shape[0], device=self.device).unsqueeze(0)
                    send_msg = torch.cat([seq_len, new_sample_token, correct_ids_index_path], dim=-1).to(torch.int)
                    self.inference_data.add_acc(seq_len.item())

            # 4. isend message including index info to drafter
            color_print(f" send msg is {send_msg}",2)
            send_verified_index_buffer[0:send_msg.shape[0]].copy_(send_msg)
            # dist.send(send_msg, dst=Config.DRAFTER_RANK)
            with self.inference_data.communication_timer:
                dist.send(send_verified_index_buffer, dst=Config.DRAFTER_RANK)
            # currently send a 1-d tensor todo to align the dim with the receiver
            # after the last communication , check if to end. 一定在这个位置 check，要不然会不正常结束
            if self.tokenizer.eos_token_id in token_ids.tolist():
                break
            # 5.  rollback kv_cache 不能算上 新 sample 出来的token
            model.rollback(output_token_ids.shape[1] - 1)
            color_print(
                f"tokens/s is {(output_token_ids.shape[1] - init_len) / (time.perf_counter() - start)}\n output_len is {output_token_ids.shape[1]}",
                5)
        self.inference_data.get_inference_data(is_store=True,
                                               file_path='/home/TreeDecoding/benchmark/exp/test/test.jsonl')
        print(f"outputs are {self.tokenizer.decode(output_token_ids[0])}")
        return output_token_ids

    def _rollback_kv_cache(self,
                           correct_ids_index_path: torch.Tensor,
                           nodes_per_layer: int,
                           is_reject: bool = False) -> None:
        # 这个位置被拒绝之后不一定长度为1
        start = self.verified_len

        tokens_len = correct_ids_index_path.size(0)
        if tokens_len == 0:
            self._kv_cache_truncate(start)
            return
        indices = torch.arange(correct_ids_index_path.numel(), device=correct_ids_index_path.device,
                               dtype=correct_ids_index_path.dtype)
        indices = start + correct_ids_index_path + indices * nodes_per_layer
        if not is_reject:
            rest_kv = torch.arange(start + nodes_per_layer * tokens_len, self.past_key_values[0][0].shape[2],
                                   device=indices.device)
            indices = torch.cat([indices, rest_kv], dim=0)
        for data in self.past_key_values_data_list:
            tgt = data[..., indices.to(data.device), :]
            dst = data[..., start: start + indices.size(0), :]
            dst.copy_(tgt, non_blocking=True)
        self.current_length_data.fill_(start + indices.size(0))
        # 更新 verified_len
        self.verified_len += tokens_len

    def _kv_cache_truncate(self, cache_len: int):
        self.current_length_data.fill_(cache_len)

    # 传入正确的 路径，更新树并 进行 kv-cache的回滚
    # 由目标模型传过来的正确的更新序列
    def _verified_update(self, tree: Tree,
                         correct_ids_index_path: torch.Tensor,
                         outputs) -> torch.Tensor:
        # 规定 在 tensor 第一个位置为标志位
        # 标志位 index[0] > 0:index[0]为验证序列长度
        # 标志位 index[1] != 0 ： 序列被拒绝， 注意，当序列被拒绝不一定长度为1 且该值为 拒绝后的token_id
        color_print(f"correct_ids_index_path is {correct_ids_index_path}",3)
        length = int(correct_ids_index_path[0].item())
        flag = int(correct_ids_index_path[1].item())
        # 正确的 token ids  index 序列
        correct_ids_index_path = correct_ids_index_path[2:length + 2]
        # if not hasattr(self, "drafter_profile_list"):
        #     self.drafter_profile_list = list()
        # self.drafter_profile_list.extend(tree.pick_path_for_profile(correct_ids_index_path))
        # if flag != -1:
        #     self.drafter_profile_list.append(flag)
        # color_print(f"drafter verified tokens are {self.tokenizer.decode(self.drafter_profile_list)}",5)

        if flag != -1:
            # 先回滚，然后携带正确 token 正常推理一次，更新树
            self._rollback_kv_cache(correct_ids_index_path, tree.nodes_per_layer, is_reject=True)
            input_id = torch.tensor(flag,dtype=torch.int, device=self.device).unsqueeze(0).unsqueeze(0)
            color_print(f"被拒绝后进行推理的 tokenid is{input_id}",3)
            outputs = self.model(
                input_ids=input_id,
                past_key_values=self.past_key_values,
            )
            # 此时verified_update 在tree会跳过一层layer
            tree.verified_update(correct_ids_index_path, is_reject=True)
            self.verified_len += 1
            return outputs

        # 进行 kv——cache 回滚
        self._rollback_kv_cache(correct_ids_index_path, tree.nodes_per_layer)
        # 更新树
        is_having_zero_weight = tree.verified_update(correct_ids_index_path)
        if is_having_zero_weight is not None:
            parent_id = is_having_zero_weight
            # 1. kv-cache truncated
            length = self.verified_len + tree.nodes_per_layer * tree.size
            self._kv_cache_truncate(length)
            # 2. enqueue using level_2_cache
            color_print(f"using level_2_cache enqueue, parent id is {parent_id}")
            input_ids, position_ids, tree_attention_mask, parents = tree.enqueue_using_level2cache(parent_id)
            # 3. send message
            # I don't send message I will change the communication method
            # 4. decode once and get output
            tree_attention_mask = self.process_tree_mask(tree_attention_mask, self.verified_len)
            if self.is_llama:
                outputs = self.model(
                    input_ids=input_ids,
                    tree_attention_mask=tree_attention_mask,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids,
                )
            if self.is_qwen:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=tree_attention_mask,
                    past_key_values=self.past_key_values,
                    cache_position=position_ids,
                )

            color_print("divergence")
        return outputs
