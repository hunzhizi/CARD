from typing import Tuple

import torch


class Tree:
    def __init__(self,
                 verified_pos_len: int,
                 device: torch.device,
                 nodes_per_layer: int = 20,
                 max_depth: int = 50,
                 ):
        self.nodes_per_layer = nodes_per_layer
        self.device = device
        self.kv_mask = torch.zeros([nodes_per_layer, 0], dtype=torch.int8, device=device)
        self.tri = torch.eye(nodes_per_layer, dtype=torch.int8, device=device)
        self.position_id = torch.zeros([nodes_per_layer], dtype=torch.long, device=device)
        self.kv_cache_mask = torch.zeros([nodes_per_layer, nodes_per_layer], dtype=torch.int8, device=device)
        self.logits_buffer: torch.Tensor = torch.zeros([max_depth, nodes_per_layer], device=device)
        self.weight_buffer: torch.Tensor = torch.zeros([max_depth, nodes_per_layer], dtype=torch.float64, device=device)
        self.input_ids_buffer: torch.Tensor = torch.zeros([max_depth, nodes_per_layer], dtype=torch.int, device=device)
        self.parents_index: torch.Tensor = torch.zeros([max_depth, nodes_per_layer], dtype=torch.int, device=device)
        self.rows: torch.Tensor = torch.arange(nodes_per_layer, device=self.device)
        self.buffer_capacity: int = max_depth
        self.head: int = 0
        self.tail: int = 0
        self.size: int = 0

        self.verified_pos_len = verified_pos_len
        # 最佳 candidate path
        self.best_candidates = list()
        self.chosen_id = -1  # 用于保存当cache全都被命中后的 logits 的选择，或者当被拒绝后重新选择的logits的 id

    def set_device(self, device: torch.device):
        self.device = device

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.buffer_capacity

    def update_tail(self) -> None:
        # 更新 tail 的 index 并维护 桶的大小
        self.tail = (self.tail + 1) % self.buffer_capacity
        self.size += 1

    def enqueue(self,
                logits: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        if self.is_full():
            raise RuntimeError(f"buffer_capacity: {self.buffer_capacity} is not enough")
        if self.is_empty():
            # 第一次推理只取最后一个 token的 logits，当被拒绝或者全部接受后更新，选择self.chosen_id 通过这个变量进行维护
            logits, ids = torch.topk(logits[0][self.chosen_id], k=self.nodes_per_layer, dim=-1)
            self.logits_buffer[self.tail].copy_(logits)
            self.weight_buffer[self.tail].copy_(logits)
            self.input_ids_buffer[self.tail].copy_(ids)
            self.parents_index[self.tail].copy_(self.rows)
            self.update_tail()
            # return variables are (input_ids, position_ids, attention_mask, parent_last)
            return (ids.unsqueeze(0),
                    self.position_id + 1 + self.verified_pos_len,
                    self.tri,
                    self.rows)
        else:
            # 获取一层中的 每一个节点的logits
            logits, ids = torch.topk(logits[0], k=self.nodes_per_layer, dim=-1)
            orig_logits = logits
            last_layer_weights = self.weight_buffer[self.tail - 1].unsqueeze(1)
            logits = logits * last_layer_weights
            flat_logits, flat_ids = logits.view(-1), ids.view(-1)
            global_top_logits, global_top_idx = torch.topk(flat_logits, k=self.nodes_per_layer, dim=-1)
            input_ids = flat_ids[global_top_idx]
            parents = global_top_idx // self.nodes_per_layer
            # 做一个实验，如果第一个token 置信度很高的话就 pad掉其他token(剪枝)
            # if global_top_logits[0] / global_top_logits[1] > 30:
            #     global_top_logits[1:] = 0
            orig_logits = orig_logits.view(-1)[global_top_idx]
            self.logits_buffer[self.tail].copy_(orig_logits)
            self.weight_buffer[self.tail].copy_(global_top_logits)
            self.input_ids_buffer[self.tail].copy_(input_ids)
            self.parents_index[self.tail].copy_(parents)
            # 制作上一次推理的 kv cache
            self.kv_cache_mask[self.rows, parents] = 1
            # 根据parents 找到前面所有kv_mask 对应的tensor，与本次新形成的kv_cache_mask 进行拼接。
            # 最后与对角阵拼接，获得当前推理的掩码
            self.kv_mask = torch.cat([self.kv_mask[parents], self.kv_cache_mask], dim=1)
            attention_mask = torch.cat([self.kv_mask, self.tri], dim=1)
            # print(f"weight_buffer is {self.weight_buffer[self.tail]}")
            self.update_tail()

            return (input_ids.unsqueeze(0),
                    self.position_id + (self.size + 1) + self.verified_pos_len,
                    attention_mask,
                    parents)

    def dequeue(self, dequeue_num: int = 0) -> None:
        # 用于目标模型验证成功后更新 cache
        if self.size - dequeue_num < 0:
            raise RuntimeError(f"Buffer is empty, cannot dequeue")
        self.head = (self.head + dequeue_num) % self.buffer_capacity
        self.size -= dequeue_num

    def verified_update(self,
                        correct_ids_index_path: torch.Tensor,
                        is_reject: bool = False):
        # 更新 mask
        self.kv_mask = torch.zeros([self.nodes_per_layer, 0], dtype=torch.int8, device=self.kv_mask.device)
        self.kv_cache_mask.fill_(0)

        if is_reject is True:
            dequeue_num = self.size
            self.dequeue(dequeue_num)
            self.chosen_id = -1
            return
        else:
            dequeue_num = correct_ids_index_path.size(0)
            # 更新buffer
            self.dequeue(dequeue_num)
            # 更新 verified_pos_len
            self.verified_pos_len += dequeue_num
        if self.is_empty():
            # 更新选中的 logits 的对应的 id
            self.chosen_id = correct_ids_index_path[-1]
            return
        # 更新 logits 对验证失败的所有其他tokens 的路径进行剪枝
        # 获取验证的最后一个tensor 的 index, id 可能会重复，所以 要index才行
        parent_id = correct_ids_index_path[-1]

        index = self.head
        mask = torch.isin(self.parents_index[index], parent_id)
        # 更新对应的weight 和 logits
        self.logits_buffer[index] *= mask
        self.weight_buffer[index] = self.logits_buffer[index].clone()
        parent_id = self.input_ids_buffer[index][mask]
        for i in range(1, self.size):
            index = (i + self.head) % self.buffer_capacity  # todo index 循环求余
            mask = torch.isin(self.parents_index[index], parent_id)
            # 更新对应的weight 和 logits
            self.logits_buffer[index] *= mask
            self.weight_buffer[index] = self.logits_buffer[index] * self.logits_buffer[
                (index - 1) % self.buffer_capacity]
            parent_id = self.input_ids_buffer[index][mask]

            # skip the first one
            parents = self.parents_index[index]
            self.kv_cache_mask[self.rows, parents] = 1
            self.kv_mask = torch.cat([self.kv_mask[parents], self.kv_cache_mask], dim=1)

    def pick_path_for_test(self) -> Tuple[list, list]:
        # 随机选取接受的长度
        random_integer = torch.randint(1, self.size, (1,)).item()
        # 从后（tail）向前选择
        picked_id = list()
        picked_index = list()
        tail = (self.head + random_integer - 1) % self.buffer_capacity
        # 每次选择 概率最大的 index
        index = torch.argmax(self.logits_buffer[tail])
        # logits_buffer 有可能全0？ 所以推理可能有问题，正常target model 会返回一个以 torch.Tensor([-1, token_id])来通知草稿模型
        print(f"self.logits_buffer[tail] is {self.logits_buffer[tail]}")
        start_id = self.input_ids_buffer[tail][index].item()
        picked_id.append(start_id)
        picked_index.append(index)
        parent_id = self.parents_index[tail][index].item()

        for i, j in enumerate(range(random_integer - 1)):
            tail -= 1
            tail = tail % self.buffer_capacity
            token = self.input_ids_buffer[tail][parent_id].item()
            picked_id.append(token)
            picked_index.append(parent_id)
            parent_id = self.parents_index[tail][parent_id].item()

        return [picked_id.pop() for i in range(len(picked_id))], [picked_index.pop() for i in range(len(picked_index))]

    def update(self, logits):
        # 根据验证序列 来更新 buffer
        pass
