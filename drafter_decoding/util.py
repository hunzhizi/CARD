import os
import random
import argparse
import torch
import torch.nn.functional as F
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


