import os
import sys

from drafter_decoding.Config import Config

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import random
from drafter_decoding.util import seed_everything, parse_arguments
from drafter_decoding.DecodingModel import DecodingModel
import torch.distributed as dist



def read_results(file_path):
    f = open(file_path)
    data = [json.loads(line) for line in f.readlines()]
    record = {}
    for item in data:
        if item["category"] not in record:
            record[item["category"]] = {"wall_time": [], "num_token": []}
        record[item["category"]]["wall_time"].append(item["time"])
        record[item["category"]]["num_token"].append(item["num_new_tokens"])
    return record


class EvalMGSM(DecodingModel):
    def __init__(self, parser_args):
        super().__init__(None, None, None, parser_args=parser_args)
        self.nodes_per_layer: int = parser_args.nodes_per_layer
        self.max_depth: int = parser_args.max_depth
        self.dataset_name = "MGSM"
        self.prompt = ""
        self.seed_set = set()

        # load relative resources
        self.load_data()

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading MGSM data...", 3)
        data = []
        with open(os.path.join(self.parser_args.data_path, "mgsm.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["question"])
                encode_special_token_flag = not (
                        "Llama-3.2-1B-Instruct" in self.parser_args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.parser_args.target_model)

                input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=encode_special_token_flag)
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                datum["ground_truth"] = datum["answer"]
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        text = self.prompt + "Question: " + input_text + "\n\n" + "Answer:"
        return text

    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def eval(self):
        out_path = os.path.join(self.parser_args.exp_name, f"{self.parser_args.eval_mode}_mgsm.jsonl")
        out_f = open(out_path, "a")
        for _ in range(self.parser_args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)
            for idx, datum in tqdm.tqdm(enumerate(self.data), total=len(self.data),
                                        disable=not self.is_target_model, ncols=50):
                input_ids = datum["input_ids"].to(self.device)
                torch.cuda.synchronize()
                if self.parser_args.eval_mode == "two_model":
                    if self.is_target_model:
                        start_time = time.time()
                        generate_ids = self.decoding_with_cache_profile(input_ids, self.nodes_per_layer, self.max_depth)
                        end_time = time.time()
                        # 结束后通知 drafter 结束
                        end_flag = torch.tensor(-1, device=self.model.device, dtype=torch.int)
                        dist.send(end_flag, dst=Config.DRAFTER_RANK)
                    if self.is_drafter:
                        self.draft(input_ids, self.nodes_per_layer, self.max_depth)
                elif self.parser_args.eval_mode == "single_model":
                    start_time = time.time()
                    generate_ids = self.autoregressive_decoding(input_ids)
                    end_time = time.time()
                torch.cuda.synchronize()
                dist.barrier()
                if self.is_target_model:
                    out_f.write(json.dumps(
                        {"question_id": idx, "category": datum["category"], "time": end_time - start_time,
                         "num_new_tokens": generate_ids.shape[1] - input_ids.shape[1],
                         "answer": self.tokenizer.decode(generate_ids[0, :], skip_special_tokens=True)},
                        ensure_ascii=False) + "\n")
                out_f.flush()

        out_f.close()

        record = read_results(out_path)

        total_num_token, total_wall_time = [], []

        for k in record:
            if k == "writing":
                num_tokens = torch.tensor(record[k]["num_token"][1:])
                wall_times = torch.tensor(record[k]["wall_time"][1:])
                total_num_token.extend(record[k]["num_token"][1:])
                total_wall_time.extend(record[k]["wall_time"][1:])
            else:
                num_tokens = torch.tensor(record[k]["num_token"])
                wall_times = torch.tensor(record[k]["wall_time"])
                total_num_token.extend(record[k]["num_token"])
                total_wall_time.extend(record[k]["wall_time"])

            speed = num_tokens / wall_times
            self.color_print(
                f"Generating speed of category {k}: {speed.float().mean().item():.2f} with std {speed.float().std().item()} token / second",
                2)

        total_speed = torch.tensor(total_num_token) / torch.tensor(total_wall_time)
        self.color_print(
            f"Average generating speed: {total_speed.float().mean().item()} with std {total_speed.float().std().item()} token / second",
            2)


if __name__ == "__main__":
    parser_args = parse_arguments()
    alg = EvalMGSM(parser_args)
    alg.eval()