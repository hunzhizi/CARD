import time
from typing import Tuple

import torch.nn as nn
import torch
from fastchat.data.split_long_conversation import max_length
from numpy.ma.core import divide

from .CacherManager import CacheManager
from .Tree import Tree
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from drafter_decoding.kv_cache import initialize_past_key_values
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def color_print(content: str, color_number: int = 4):
    """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
    # if self.accelerator.is_main_process:
    print(f"\033[9{color_number}m{content}\033[0m")


class DecodingModel(nn.Module):
    def __init__(self,
                 model,
                 model_name_or_path,
                 device: torch.device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.hidden_size = model.lm_head.weight.shape[-1]
        self.vocab_size = model.lm_head.weight.shape[0]
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.verified_len: int = 0
        self.device = device

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
        # drafter using the customized model
        if draft_model_path is not None:
            model = KVLlamaForCausalLM.from_pretrained(
                draft_model_path, **kwargs
            ).eval()
        # target model using the class from transformers lib on huggingface
        elif target_model_path is not None:
            model = AutoModelForCausalLM.from_pretrained(target_model_path,
                                                         **kwargs).eval()
        # cls() 调用 MyClass 的 __init__ 方法创建了一个新实例
        return cls(model, draft_model_path, device=main_device)

    def process_tree_mask(self, tree_attention_mask, init_len):
        # todo 考虑封装到 Tree 中
        attention_mask = torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask = torch.cat([attention_mask, tree_mask], dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def autogressive_decoding(self,
                              input_ids: int) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
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
        print(self.tokenizer.decode(input_ids[0]))

        # prefill 阶段
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
        )
        input_id = torch.argmax(outputs[0][:, -1, :], dim=-1)
        input_id = input_id.unsqueeze(0)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        max_length = 500
        for i in range(max_length):
            outputs = self.model(
                input_ids=input_id,
                past_key_values=past_key_values
            )
            input_id = torch.argmax(outputs[0], dim=-1)
            input_id = input_id
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            print(f"output is : {self.tokenizer.decode(input_ids[0])}")
        return input_ids

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
                tensor1 = torch.tensor(index, device='cuda')
                tensor2 = torch.tensor(len(index), device='cuda').unsqueeze(0)
                tensor3 = torch.tensor(0, device='cuda').unsqueeze(0)
                buffer = torch.cat([tensor2, tensor3, tensor1], dim=-1)
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
            # print(f"the token_ids are {debug_queue}")
            print(f"the total len is {len(debug_queue)}")
            print(f"{j}..{(time.perf_counter() - start_time)}............")
            j += 1

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
        # create tree ,start recv cache
        tree_buffer = Tree(input_ids.shape[1], self.device, nodes_per_layer, max_depth)
        # todo init distributed env
        cache_manager = CacheManager(self.world_size, self.rank, self.device, tree_buffer, is_target_model=True)
        # prefill phase
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            self.past_key_values = None
            outputs = self.model(input_ids)
            self.past_key_values = outputs.past_key_values
        else:
            outputs = self.model(input_ids)
            self.past_key_values = outputs.past_key_values
        # decode phase
        # 1. query cache
        cache_manager

    def _rollback_kv_cache(self,
                           correct_ids_index_path: torch.Tensor,
                           nodes_per_layer: int,
                           is_reject: bool = False) -> None:
        # 这个位置被拒绝之后不一定长度为1
        start = self.verified_len

        tokens_len = correct_ids_index_path.size(0)
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
        # 标志位 index[0]== -1 ： 序列被拒绝， 注意，当序列被拒绝不一定长度为1
        # 标志位 index[1] 为 被拒绝后的 token_id
        length = int(correct_ids_index_path[0].item())
        # 正确的 token ids  index 序列
        correct_ids_index_path = correct_ids_index_path[2:length + 2]

        if length == -1:
            # 先回滚，然后携带正确 token 正常推理一次，更新树
            self._rollback_kv_cache(correct_ids_index_path, tree.nodes_per_layer, is_reject=True)
            input_id = correct_ids_index_path[1].unsqueeze(0).unsqueeze(0)
            outputs = self.model(
                input_ids=input_id,
                attention_mask=None,
                tree_attention_mask=None,
                past_key_values=self.past_key_values,
                position_ids=None,
            )
            tree.verified_update(correct_ids_index_path, is_reject=True)
            self.verified_len += correct_ids_index_path.size(0)
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
            input_ids, position_ids, tree_attention_mask, parents = tree.enqueue_using_level2cache(parent_id)
            tree_attention_mask = self.process_tree_mask(tree_attention_mask, self.verified_len)
            outputs = self.model(
                input_ids=input_ids,
                tree_attention_mask=tree_attention_mask,
                past_key_values=self.past_key_values,
                position_ids=position_ids,
            )
            # 3. send message
            # 4. decode once and get output

            color_print("divergence")
        return outputs
