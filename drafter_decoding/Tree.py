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
        self.weight_buffer: torch.Tensor = torch.zeros([max_depth, nodes_per_layer],dtype=torch.float64, device=device)
        self.input_ids_buffer: torch.Tensor = torch.zeros([max_depth, nodes_per_layer], device=device)
        self.parents_index: torch.Tensor = torch.zeros([max_depth, nodes_per_layer], device=device)
        self.rows: torch.Tensor = torch.arange(nodes_per_layer, device=self.device)
        self.buffer_capacity: int = max_depth
        self.head: int = 0
        self.tail: int = 0
        self.size: int = 0

        self.verified_pos_len = verified_pos_len
        # 最佳 candidate path
        self.best_candidates = list()

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
            # 此时只取最后一个 token的 logits
            logits, ids = torch.topk(logits[0][-1], k=self.nodes_per_layer, dim=-1)
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
            last_layer_weights = self.weight_buffer[self.tail - 1].unsqueeze(1)
            logits = logits * last_layer_weights
            flat_logits, flat_ids = logits.view(-1), ids.view(-1)
            global_top_logits, global_top_idx = torch.topk(flat_logits, k=self.nodes_per_layer, dim=-1)
            input_ids = flat_ids[global_top_idx]
            parents = global_top_idx // self.nodes_per_layer
            self.weight_buffer[self.tail].copy_(global_top_logits)
            self.input_ids_buffer[self.tail].copy_(input_ids)
            self.parents_index[self.tail].copy_(parents)
            # 制作上一次推理的 kv cache
            self.kv_cache_mask[self.rows, parents] = 1
            # 根据parents 找到前面所有kv_mask 对应的tensor，与本次新形成的kv_cache_mask 进行拼接。
            # 最后与对角阵拼接，获得当前推理的掩码
            self.kv_mask = torch.cat([self.kv_mask[parents], self.kv_cache_mask], dim=1)
            attention_mask = torch.cat([self.kv_mask, self.tri], dim=1)
            print(f"weight_buffer is {self.weight_buffer[self.tail]}")
            self.update_tail()

            return (input_ids.unsqueeze(0),
                    self.position_id + (self.size + 1) + self.verified_pos_len,
                    attention_mask,
                    parents)

    def update(self, logits):
        # 根据验证序列 来更新 buffer
        pass

if __name__ == '__main__':
    result = 0.6 ** 22
    print(result)