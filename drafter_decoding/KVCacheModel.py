import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from drafter_decoding.Config import Config


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list
    return None


class KVCacheModel(nn.Module):
    def __init__(self, model: torch.nn.Module, init_input_len: int, temperature: float = 0, top_k: int = 0,
                 top_p: float = 0,
                 vocab_size: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._past_key_values = None
        self.current_verified_len = 0
        self.logits_processor = None
        if temperature > 1e-5:
            self.logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
            print(f"decoding temperature is {temperature}")

        self.sum = 0
        self.device = model.device
        self.init_input_len = init_input_len
        self.handshake_flag = torch.ones(1, device=model.device, dtype=torch.int)


    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''

        Args:
            input_ids: new input ids which don't have kv-cache

        Returns:
            verified tokens

        '''
        if self._past_key_values is None:
            tokens_id = self._generate_some_tokens_with_kvcache(input_ids,True)
        else:
            tokens_id = self._generate_some_tokens_with_kvcache(input_ids)
        return tokens_id

    def compare_tensors(self, tensor1, tensor2):
        assert len(tensor2) == len(tensor1) + 1, "tensor2的长度必须比tensor1大1"
        length = len(tensor1)
        # 直接比较前length个元素，生成差异掩码
        diff_mask = tensor1 != tensor2[:length]
        # 检查是否存在不同元素
        if not torch.any(diff_mask):
            return tensor2
        # 找到第一个不同位置的索引
        first_diff_index = torch.argmax(diff_mask.int())
        # 返回公共部分及tensor2中的不同元素
        return tensor2[:first_diff_index + 1]

    def _generate_some_tokens_with_kvcache(self,
                                           prefix: torch.Tensor,
                                           is_prefill: bool = False) -> torch.Tensor:
        """

        :param prefix (torch.Tensor): the prefix
        :return:Torch.Tensor: verified tokens 1D tensor
                bool: is accepted all tokens?
        """
        new_tokens: torch.FloatTensor = self._forward_with_kvcache(prefix)
        if is_prefill:
            # only support for bs == 1
            prefix = prefix[:, self.init_input_len:]
            # new_tokens = new_tokens[0, self.init_input_len:]
        else:
            prefix = prefix[:,1:]
        # verify the new_tokens
        return self.compare_tensors(prefix[0], new_tokens[0])

    @torch.no_grad()
    def rollback(self, end_pos: int):
        assert self._past_key_values is not None, "past_key_values is None"

        self._past_key_values = [
            (k[..., :end_pos, :], v[..., :end_pos, :])
            for k, v in self._past_key_values
        ]
        self.current_verified_len = end_pos

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.FloatTensor:
        # 第一次推理没有保存kvcache ，此时调用forward
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            # send msg to get tree info asynchronously
            # 这里其实不太符合开闭原则,但是为了 asynchronously 最好是在这里进行tree info 的获取
            # dist.isend(self.handshake_flag,dst=Config.DRAFTER_RANK)

            # logit shape is (batch_size, sequence_length, vocab_size)
            if (outputs.logits.dim() == 2):
                outputs.logits = outputs.logits.unsqueeze(0)
            seq_len = outputs.logits.size(1)
            # 记录kvcache
            self._past_key_values = outputs.past_key_values
            logits = outputs.logits[:, self.init_input_len - 1:, :]
        else:
            # 有kvcache 进行的推理
            # return the last token's logits
            # 注意： 这里的 seq_len 不是input_ids 的len 是含有kvcache的 seq len 不包含上次cat 的tokens
            # seq_len = self._past_key_values[0][0].shape[2]
            # caution 这里获取 seq_len 并不是脱裤子放屁，很重要的一步操作，因为seq_len 后面有一些tokens被接受了，但是没有kvcache
            # new_input_id = input_ids[:, seq_len:]
            new_input_id = input_ids
            # 保证 input_id.dim() == 2
            if new_input_id.dim() == 1:
                new_input_id = torch.unsqueeze(new_input_id, 0)

            # 进行推理，传入当前 token_id 和 past_key_values ,使用use_cache 进行推理
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            outputs = self._model(input_ids=new_input_id,
                                  past_key_values=self._past_key_values,
                                  use_cache=True)
            # end_event.record()
            # torch.cuda.synchronize()
            # self.sum += start_event.elapsed_time(end_event) / 1000
            # print(f"generate time is {self.sum}")

            # send msg to get tree info asynchronously
            # 这里其实不太符合开闭原则,但是为了 asynchronously 最好是在这里进行tree info 的获取
            # dist.isend(self.handshake_flag,dst=Config.DRAFTER_RANK)
            logits = outputs.logits

            if logits.dim() == 2:
                logits = torch.torch.unsqueeze(logits, 0)

        if self.logits_processor is None:
            new_tokens = torch.argmax(logits, dim=-1)
            # print(f" new tokens are {new_tokens}")
        else:
            logits = self.logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            new_tokens = torch.multinomial(probabilities, 1).view(1, -1)
            # print(f" new tokens are logits_processor {new_tokens}")
            # self.current_verified_len += outputs.logits.size(1)

            # last_token_logits = logits[:, -1, :]
        self._past_key_values = outputs.past_key_values

        return new_tokens
