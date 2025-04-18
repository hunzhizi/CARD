import time

import torch.nn as nn
import torch
from numpy.ma.core import divide

from .Tree import Tree
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from drafter_decoding.kv_cache import initialize_past_key_values
from transformers import AutoTokenizer


class DecodingModel(nn.Module):
    def __init__(self,
                 model,
                 model_name_or_path,

                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.hidden_size = model.lm_head.weight.shape[-1]
        self.vocab_size = model.lm_head.weight.shape[0]
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tree = None
        self.verified_len: int = 0

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            draft_model_path=None,
            **kwargs,
    ):

        draft_model = KVLlamaForCausalLM.from_pretrained(
            draft_model_path, **kwargs
        )
        # cls() 调用 MyClass 的 __init__ 方法创建了一个新实例
        return cls(draft_model, draft_model_path)

    def process_tree_mask(self, tree_attention_mask, init_len):
        # todo 考虑封装到 Tree 中
        attention_mask = torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask = torch.cat([attention_mask, tree_mask], dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    # draft 是一个不断进行树状起草的函数，采用预先分配的kv cache 进行起草工作
    @torch.no_grad()
    def draft(self,
              input_ids: int,
              nodes_per_layer: int = 20,
              max_depth: int = 50,
              ):
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
        tree = Tree(verified_len_posi, outputs[0].device, nodes_per_layer=5)
        # decode 阶段，在 decode 阶段，只要没有被完全拒绝，每一次都要处理一层的树节点。
        # 通过 Tree 类进行管理
        tokens_id = []

        input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
            torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
        tree_attention_mask = self.process_tree_mask(tree_attention_mask, init_len)
        print(self.tokenizer.decode(input_ids[0]))
        tokens_id.append(input_ids[0][0].item())
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
            if len(tokens_id) == 4:
                self._verified_update(tree, torch.tensor([3, 0, 0, 0], dtype=torch.int32, device='cuda'))
                tokens_id.pop()
                tokens_id.pop()
                tokens_id.pop()
            # outputs[0]： 表示的是 logits
            # 第一次推理节点的维度为 size(1,23,32000)
            # size(batch_size, seq_len, vocab_size)
            # 可以根据 seq_len 来判断当前处理情况是一层树，还是只是正常推理
            input_ids, position_ids, tree_attention_mask, parents = tree.enqueue(
                torch.softmax(outputs[0], dim=-1, dtype=torch.float32))
            if outputs[0].size(1) != 1:
                tree_attention_mask = self.process_tree_mask(tree_attention_mask, self.verified_len)

            tokens_id.append(input_ids[0][0].item())
            print(self.tokenizer.decode(input_ids[0]))
            print(f"the best candidates are \n{self.tokenizer.decode(tokens_id)}")
            print(f"the total len is {len(tokens_id)}")
            print(f"{j}..{(time.perf_counter() - start_time)}............")
            j+=1


    def _rollback_kv_cache(self,
                           correct_ids_index_path: torch.Tensor,
                           nodes_per_layer: int):
        start = self.verified_len
        tokens_len = correct_ids_index_path.size(0)
        indices = torch.arange(correct_ids_index_path.numel(), device=correct_ids_index_path.device,
                               dtype=correct_ids_index_path.dtype)
        indices = start + correct_ids_index_path + indices * nodes_per_layer
        rest_kv = torch.arange(start + nodes_per_layer * tokens_len,self.past_key_values[0][0].shape[2],device=indices.device)
        indices = torch.cat([indices, rest_kv],dim=0)
        for data in self.past_key_values_data_list:
            tgt = data[..., indices.to(data.device), :]
            dst = data[..., start: start + indices.size(0), :]
            dst.copy_(tgt, non_blocking=True)
        self.current_length_data.fill_(start + indices.size(0))
        # 更新 verified_len
        self.verified_len += tokens_len

    # 传入正确的 路径，更新树并 进行 kv-cache的回滚
    # 由目标模型传过来的正确的更新序列
    def _verified_update(self, tree: Tree, correct_ids_index_path: torch.Tensor):
        # 规定 在 tensor 第一个位置为 验证序列长度
        length = int(correct_ids_index_path[0].item())
        # 正确的 token ids  index 序列
        correct_ids_index_path = correct_ids_index_path[1:length + 1]

        # 进行 kv——cache 回滚
        self._rollback_kv_cache(correct_ids_index_path, tree.nodes_per_layer)
        # 更新树
        tree.verified_update(correct_ids_index_path)

