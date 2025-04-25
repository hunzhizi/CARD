import threading
from threading import Thread

import torch
import torch.distributed as dist

from drafter_decoding.Config import Config
from drafter_decoding.Tree import Tree


class CacheManager:
    def __init__(self,
                 world_size: int,
                 rank: int,
                 device,
                 tree: Tree ,
                 is_drafter: bool = False,
                 is_calibration: bool = False,
                 is_target_model: bool = False):
        # 检查是否恰好有一个布尔值为True
        true_count = sum([is_drafter, is_calibration, is_target_model])
        if true_count != 1:
            raise ValueError("必须且只能有一个布尔属性为True")

        self.is_drafter = is_drafter
        self.is_calibration = is_calibration
        self.is_target_model = is_target_model
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.tree_buffer = tree
        self.nodes_per_layer = tree.nodes_per_layer
        if self.is_drafter:
            self.is_update = False
            # 第一个位置作为标志位，最长长度支持到 tree.buffer_capacity
            self.recv_buffer = torch.zeros(tree.buffer_capacity + 2, device=self.device)
        # 开启线程
        self.lock = threading.Lock()
        # _get_recv_thread() needs init lock first
        self._get_recv_thread()

    def _get_recv_thread(self) -> None:
        '''
        recv_thread :用于接收别的进程传过来的参数，并组织进入 tree_buffer 中
        Returns:

        '''
        if self.is_drafter:
            # it depends on the world_size ,
            # if world_size = 2 , only drafter and target model
            # if world_size = 3 , drafter and calibration model and target model
            if self.world_size == 2:
                def recv_method() -> None:
                    # 接收 target model 传过来的验证后的 id
                    # 置更改标志位为 True
                    while True:
                        dist.recv(self.recv_buffer, src=Config.TARGET_MODEL_RANK)
                        self.is_update = True  # 可能有线程安全， 但是概率极低

                thread = threading.Thread(
                    target=recv_method,
                )
                thread.daemon = True
                thread.start()
                # thread.join()

            elif self.world_size == 3:
                pass
            pass
            return None
        elif self.is_calibration:
            pass
        elif self.is_target_model:
            # 目标模型需要
            # 1. 接收 每一层的树
            # 2. 做解包
            # 3. 更新
            if self.world_size == 2:
                def recv_method():
                    buffer_combination = torch.zeros(self.nodes_per_layer * 4, dtype=torch.float32, device=self.device)
                    while True:
                        dist.recv(tensor=buffer_combination, src=Config.DRAFTER_RANK)
                        # 接收信息之后放入 tree_buffer 中
                        print(f"taraget recv mes {buffer_combination}")
                        with self.lock:
                            self.tree_buffer.update_cache_for_target_model(buffer_combination)

                thread = threading.Thread(
                    target=recv_method,
                )
                thread.daemon = True
                thread.start()
                # thread.join()

            elif self.world_size == 3:
                pass
            pass


    def query_cache(self):
        '''用于进行tree_buffer 的cache 选择 for target model and calibration dataset'''
        return self.tree_buffer.get_candidates_for_target_model()


    def update_tree_buffer(self,
                           correct_ids_index_path: torch.Tensor,
                           new_sample_token_id: torch.Tensor,
                           root_index: int
                           ) -> torch.Tensor | bool:
        with self.lock:
            return self.tree_buffer.verified_update_for_target_model(correct_ids_index_path, new_sample_token_id, root_index)