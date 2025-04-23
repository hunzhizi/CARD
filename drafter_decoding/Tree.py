from operator import index
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
        self.pending_tail: int = 0 # 用于 在 verified_update 过程中 后续weight 更新为0 此时未命中，为了保证通信同步而设置的变量
        # level-2 cache for updating buffer when weight is all zero
        self.level2cache_input_ids: torch.Tensor = torch.zeros([max_depth,nodes_per_layer,nodes_per_layer], device=device)
        self.level2cache_logits: torch.Tensor = torch.zeros([max_depth,nodes_per_layer,nodes_per_layer], device=device)

        self.verified_pos_len = verified_pos_len
        # 最佳 candidate path
        self.best_candidates:list = list()
        self.chosen_id = -1  # 用于保存当cache全都被命中后的 logits 的选择，或者当被拒绝后重新选择的logits的 id
        self.parent_id = 0 # 用于维护 update_for_target_model 中的 parent_id

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
            # 注意存放的位置是在当前layer 去找，不是上一层。
            self.level2cache_logits[self.tail] = logits
            self.level2cache_input_ids[self.tail] = ids
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
            self.kv_cache_mask.zero_()
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


    def enqueue_using_level2cache(self,parent_id):
        # index = (self.tail - 1) % self.buffer_capacity
        index = self.tail
        # make a mask
        mask = torch.zeros(self.nodes_per_layer,device=self.device)
        mask[parent_id] = True
        mask = mask.unsqueeze(1)
        flat_ids = (self.level2cache_input_ids[index] * mask).to(torch.int).view(-1)
        flat_logits = (self.level2cache_logits[index] * mask) .view(-1)
        global_top_logits, global_top_idx = torch.topk(flat_logits, k=self.nodes_per_layer, dim=-1)
        input_ids = flat_ids[global_top_idx]
        parents = global_top_idx // self.nodes_per_layer
        logits = flat_logits[global_top_idx]
        self.logits_buffer[self.tail].copy_(logits)
        self.weight_buffer[self.tail].copy_(global_top_logits)
        self.input_ids_buffer[self.tail].copy_(input_ids)
        self.parents_index[self.tail].copy_(parents)
        if self.is_empty():
            self.update_tail()
            return (input_ids.unsqueeze(0),
                    self.position_id + 1 + self.verified_pos_len,
                    self.tri,
                    self.rows)

        # 制作上一次推理的 kv cache
        self.kv_cache_mask.zero_()
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

    def truncate_path(self, index: int):
        self.tail = index
        self.size = (self.tail - self.head )% self.buffer_capacity
        self.pending_tail = index

    def verified_update(self,
                        correct_ids_index_path: torch.Tensor,
                        is_reject: bool = False) -> torch.Tensor | None:
        # 更新 mask
        self.kv_mask = torch.zeros([self.nodes_per_layer, 0], dtype=torch.int8, device=self.kv_mask.device)
        # self.kv_cache_mask.fill_(0)

        dequeue_num = correct_ids_index_path.size(0)
        # 更新buffer
        self.dequeue(dequeue_num)
        # 更新 verified_pos_len
        self.verified_pos_len += dequeue_num
        if is_reject is True:
            # 将后面的 buffer 清零
            self.head = self.tail
            self.size = 0
            return False
        # if self.is_empty():
        #     # 更新选中的 logits 的对应的 id
        #     self.chosen_id = correct_ids_index_path[-1]
        #     return
        # 更新 logits 对验证失败的所有其他tokens 的路径进行剪枝
        # 获取验证的最后一个tensor 的 index, id 可能会重复，所以 要index才行
        parent_id = correct_ids_index_path[-1]

        index = self.head
        mask = torch.isin(self.parents_index[index], parent_id)
        # 检查 mask 是否全为 0 if mask == 0 表明后续无正确 cache，需要回滚更新
        if torch.sum(mask).item() == 0:
            # todo
            self.truncate_path(index)
            print(f"更新失败")
            return parent_id
        # 更新对应的weight 和 logits
        self.logits_buffer[index] *= mask
        self.weight_buffer[index] = self.logits_buffer[index].clone()
        parent_id = torch.nonzero(mask).view(-1)
        for i in range(1, self.size):
            index = (i + self.head) % self.buffer_capacity
            mask = torch.isin(self.parents_index[index], parent_id)
            if torch.sum(mask).item() == 0:
                self.truncate_path(index)
                print(f"更新失败")
                return parent_id
            # 更新对应的weight 和 logits
            self.weight_buffer[index] = self.weight_buffer[(index - 1) % self.buffer_capacity][self.parents_index[index]] * self.logits_buffer[index]
            parent_id = torch.nonzero(mask).view(-1)

            # skip the first one
            parents = self.parents_index[index]
            self.kv_cache_mask.zero_()
            self.kv_cache_mask[self.rows, parents] = 1
            self.kv_mask = torch.cat([self.kv_mask[parents], self.kv_cache_mask], dim=1)
        return None

    def pick_path_for_test(self) -> Tuple[list, list]:
        # 随机选取接受的长度
        random_integer = torch.randint(1, self.size, (1,)).item()

        picked_index = list()
        best_candidate = list()
        head = self.head
        weights = self.weight_buffer[head].clone()
        for i in range(random_integer):
            parent_id = torch.argmax(weights)
            picked_index.append(parent_id)
            best_candidate.append(self.input_ids_buffer[head][parent_id].item())
            head += +1
            head %= self.buffer_capacity
            mask = torch.isin(self.parents_index[head], parent_id)
            if torch.sum(mask).item() == 0:
                print(f"选择路径失败")
                break
            weights = self.weight_buffer[head].clone()
            weights*=mask
            if torch.sum(weights).item() == 0:
                break

        return best_candidate,picked_index

        # return [picked_id.pop() for i in range(len(picked_id))], [picked_index.pop() for i in range(len(picked_index))]

    def update_for_target_model(self, combination_buffer: torch.Tensor):
        '''
        to maintain the tree for target model
        Args:
            combination_buffer: combination of received buffer

        Returns:

        '''
        # 1. 做解包
        # 2. 更新
        offset = self.nodes_per_layer
        self.logits_buffer[self.tail] = combination_buffer[:offset]
        self.weight_buffer[self.tail] = combination_buffer[offset:2 * offset].to(torch.float64)
        self.input_ids_buffer[self.tail] = combination_buffer[offset * 2: offset * 3].to(torch.int)
        self.parents_index[self.tail] = combination_buffer[offset * 4 :].to(torch.int)
        self.update_tail()



    def get_candidates_for_target_model(self):
        # todo 未完成
        best_candidate = list()
        # 从head 开始进行选取
        # 更新完后检查 weight_buffer 是否为 0 ，在 verify 阶段检查？ 还是在这里进行检查？
        # 在 verify 阶段检查 weight_buffer 全为0 的情况保证有效性
        parent_id = torch.argmax(self.weight_buffer[self.head])
        best_candidate.append(self.input_ids_buffer[parent_id])


