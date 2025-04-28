import threading
from threading import Thread

import torch
import torch.distributed as dist

from drafter_decoding.Config import Config
from drafter_decoding.Tree import Tree

def color_print(content: str, color_number: int = 4):
    """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
    # if self.accelerator.is_main_process:
    print(f"\033[9{color_number}m{content}\033[0m")

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
            self.recv_buffer = torch.zeros(tree.buffer_capacity + 2, device=self.device ,dtype=torch.int)
        # 开启线程
        self.handshake_flag = torch.zeros(1,device=device,dtype=torch.int)
        self.combination_buffer = torch.zeros(tree.buffer_capacity,tree.nodes_per_layer * 4 + 1, device=self.device)
        self.work = None
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
                        # 首先进行握手通知线程进行信息拉去
                        color_print(f"_get_recv_thread 准备get recv msg {self.handshake_flag}")
                        dist.recv(self.handshake_flag, src=Config.TARGET_MODEL_RANK)
                        color_print(f"_get_recv_thread get recv msg {self.handshake_flag}")
                        # handshake_flag == 0 表示 单纯拉取 target model 第一次进行query cache 此时没有验证信息，只需要拉去cache,
                        # 或者当target model 被拒绝后再一次拉去信息
                        # handshake_flag == 1 表示 握手通讯
                        # todo with lock
                        if self.tree_buffer.size >= Config.PREDICTION_NUM and self.is_update == False:
                            color_print(f"_get_recv_thread 准备 send tree buffer state")
                            send_msg = self.tree_buffer.get_send_msg_for_drafter()
                            dist.send(self.tree_buffer.get_send_msg_for_drafter(), Config.TARGET_MODEL_RANK)
                            color_print(f"_get_recv_thread send tree buffer state {send_msg}")
                        else:
                            color_print(f"_get_recv_thread 获取锁")
                            with self.tree_buffer.global_condition:
                                self.tree_buffer.global_condition.wait()
                                color_print(f"_get_recv_thread 准备 send tree buffer state")
                                send_msg = self.tree_buffer.get_send_msg_for_drafter()
                                dist.send(self.tree_buffer.get_send_msg_for_drafter(), Config.TARGET_MODEL_RANK)
                            color_print(f"_get_recv_thread send tree buffer state {send_msg}")
                        if self.handshake_flag == 0:
                            color_print(f"handshake_flag is {self.handshake_flag}")
                            continue
                        else:
                            color_print(f"handshake_flag is {self.handshake_flag}, 准备 recv msg from target model")
                            dist.recv(self.recv_buffer, src=Config.TARGET_MODEL_RANK)
                            self.is_update = True  # 可能有线程安全， 但是概率极低
                            color_print(f" recv msg from target model\n recv_buffer is {self.recv_buffer}")

                thread = threading.Thread(
                    target=recv_method,
                )
                thread.daemon = True
                thread.start()

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
            # if self.world_size == 2:
            #     def recv_method():
            #         buffer_combination = torch.zeros(self.nodes_per_layer * 4, dtype=torch.float32, device=self.device)
            #         while True:
            #             print(f"target model start recv from src={Config.DRAFTER_RANK}, self.rank = {self.rank}")
            #             dist.recv(tensor=buffer_combination, src=Config.DRAFTER_RANK)
            #             # 接收信息之后放入 tree_buffer 中
            #             with self.lock:
            #                 self.tree_buffer.update_cache_for_target_model(buffer_combination)
            #                 print(f"target model recv msg")
            #                 if self.tree_buffer.size >= Config.PREDICTION_NUM:
            #                     with self.global_condition:
            #                         self.global_condition.notify_all()
            #
            #     thread = threading.Thread(
            #         target=recv_method,
            #     )
            #     thread.daemon = True
            #     thread.start()

            # elif self.world_size == 3:
            #     pass
            pass

    def recv_buffer_for_target_model(self):
        self.work = dist.irecv(self.combination_buffer,src=Config.DRAFTER_RANK)

    def update_cache_for_target_model(self):
        self.work.wait()
        print(f"target model 接收到 来自 drafter 的 tree state {self.combination_buffer}")
        self.tree_buffer.update_cache_for_target_model(self.combination_buffer)

    def query_cache(self):
        '''用于进行tree_buffer 的cache 选择 for target model and calibration dataset'''
        return self.tree_buffer.get_candidates_for_target_model()


    def update_tree_buffer(self,
                           correct_ids_index_path: list,
                           new_sample_token_id: torch.Tensor,
                           root_index: int
                           ) -> torch.Tensor | bool:
        return self.tree_buffer.verified_update_for_target_model(correct_ids_index_path, new_sample_token_id, root_index)