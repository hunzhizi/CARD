import os
import random
import argparse
import torch
import json
from typing import Tuple
from time import perf_counter
import numpy as np

from drafter_decoding.Config import Config


def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def model_map(args):
    vocab_size = {
    "Qwen2.5-0.5B-Instruct": 151936,
    "Qwen2.5-1.5B-Instruct": 151936,
    "Qwen2.5-3B-Instruct": 151936,
    "Qwen2.5-7B-Instruct": 151936,
    "Llama-3.2-1B-Instruct":128256,
    "Llama-3.2-3B-Instruct":128256,
    "Llama-3.1-8B-Instruct":128256,
    "Llama-3.1-70B-Instruct":128256,

    }

    model_dir_map = {
        "Qwen2.5-7B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-7B-Instruct",
        "Qwen2.5-1.5B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-3B-Instruct",
        "Qwen2.5-0.5B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-0.5B-Instruct",
        "Llama-3.2-1B-Instruct": f"{Config.MODEL_DIR}/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct": f"{Config.MODEL_DIR}/Llama-3.2-3B-Instruct",
        "Llama-3.1-8B-Instruct": f"{Config.MODEL_DIR}/Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct": f"{Config.MODEL_DIR}/Llama-3.1-70B-Instruct",
    }
    # caution: all the models' vocab size should be the same
    args.draft_models_dir = [model_dir_map[model_name] for model_name in args.draft_models]
    args.target_model_dir = model_dir_map[args.target_model]
    if args.model_name is not None and args.model_name != "":
        # 作为 target model 进行测试
        print(f"args.model is {args.model_name}")
        args.target_model_dir = model_dir_map[args.model_name]


def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--data_path', type=str, default="../data")

    parser.add_argument('--draft_models', type=str, nargs='+', default=["Llama-3.2-1B-Instruct"])
    parser.add_argument('--target_model', type=str, default="Llama-3.1-8B-Instruct")

    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    # 实验相关
    # todo add more eval mode
    parser.add_argument('--eval_mode', type=str, default="default",
                        choices=["two_model", "three_model", "single_model"], help='eval mode.')

    parser.add_argument("--model_name", type=str, default="",
                        help="when '--eval_mode' is single_model ,use this to specify the model name.")

    parser.add_argument('--num_samples_per_task', '-n', type=int, default=1,
                        help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234,
                        help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max token number generated.')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    # 框架设置相关
    parser.add_argument("--branch-prediction-num", type=int, default=2, help="branch prediction number for smallest drafter.")
    parser.add_argument("--nodes_per_layer", type=int, default=20, help="tree buffer's nodes number per layer.")
    parser.add_argument("--max_depth", type=int, default=50, help="tree buffer's max depth")

    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_map(args)
    # if args.eval_mode == "para_sd":
    args.rank = int(os.environ["RANK"])  # 自动从环境变量获取
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    return args


def save_dict_to_jsonl(data: dict, file_path: str):
    try:
        # 获取文件所在的目录
        dir_path = os.path.dirname(file_path)
        # 如果目录不存在，则创建目录
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "a", encoding="utf-8") as file:
            json_line = json.dumps(data, ensure_ascii=False)
            file.write(json_line + '\n')
    except Exception as e:
        print(f"保存文件的时候出现错误:{e}")



class InferenceData:
    """
    用于存储推理相关的信息
    注意在使用过程中要记录三个信息才能进正确使用该模块
    分别是：
    acc_len_list:list
    reject_len_list
    generate_timer
    """
    def __init__(self, rank: int = 0):
        self.rank: int = rank
        # 注意 这里的 acc_len 只包含 新生成的 token 原本大模型产生的token 不包含在内
        self.acc_len_list: list = list()
        self.reject_times: int = 0 # record reject times
        self.forward_times: int = 0
        self.exe_time: float = 0
        self.generate_timer = self.Timer()
        self.verification_timer = self.Timer()
        self.communication_timer = self.Timer()


    def reset_data(self):
        self.acc_len_list.clear()
        self.reject_times = 0
        self.forward_times = 0
        self.exe_time = 0
        self.generate_timer.time_list.clear()
        self.verification_timer.time_list.clear()
        self.communication_timer.time_list.clear()

    def to_dict(self)-> dict:
        return self.__dict__

    def _get_tokens_per_second_and_mean_acc_len(self) -> Tuple[float,float]:
        if self.exe_time == 0:
            raise RuntimeError(f"you need to set exe_time,current exe_time is{self.exe_time}")
        all_acc_tokens = [i + 1 for i in self.acc_len_list]
        generate_tokens_num = sum(all_acc_tokens)
        self.forward_times = len(all_acc_tokens)
        mean_acc_len = generate_tokens_num/len(all_acc_tokens)
        return generate_tokens_num/self.exe_time, mean_acc_len

    def add_acc(self,acc_num: int):
        self.acc_len_list.append(acc_num)

    def add_reject(self):
        self.reject_times += 1


    def get_inference_data(self,
                           is_store: bool = False,
                           is_reset: bool = True,
                           file_path: str = None):
        generate_time = self.generate_timer.get_sum_time()
        verification_time = self.verification_timer.get_sum_time()
        communication_time = self.communication_timer.get_sum_time()
        self.exe_time = generate_time + verification_time + communication_time
        if verification_time == -1:
            self.exe_time += 1
        if communication_time == -1:
            self.exe_time += 1
        tokens_per_sec,mean_acc_len = self._get_tokens_per_second_and_mean_acc_len()
        data_view: dict = {
            "self.rank": self.rank,
            "tokens_per_sec":tokens_per_sec,
            "mean_acc_len": mean_acc_len,
            "exe_time": self.exe_time,
            "generate_time": generate_time,
            "verification_time": verification_time,
            "communication_time": communication_time,
            "forward_time": self.forward_times,
            "reject_len_list": self.reject_times,
            "acc_len_list": self.acc_len_list,
        }
        if is_store:
            print(data_view)
            save_dict_to_jsonl(data_view,file_path=file_path)
        if is_reset:
            self.reset_data()
        return data_view

    class Timer:
        def __init__(self):
            self.time_list: list = list()


        def __enter__(self):
            self.start_time = perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = perf_counter()
            self.execution_time = self.end_time - self.start_time
            self.time_list.append(self.execution_time)

        def get_sum_time(self) -> float:
            if len(self.time_list) == 0:
                # -1 表示没有记录该项
                return -1
            return sum(self.time_list)
