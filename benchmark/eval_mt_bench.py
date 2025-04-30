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
from fastchat.model import get_conversation_template
import shortuuid



def read_results(file_path):
    f = open(file_path)
    data = [json.loads(line) for line in f.readlines()]
    record = {}
    for item in data:
        if item["category"] not in record:
            record[item["category"]] = {"wall_time": [], "num_token": []}
        for choice in item["choices"]:
            record[item["category"]]["wall_time"].extend(choice["wall_time"])
            record[item["category"]]["num_token"].extend(choice["num_token"])
    return record


class EvalMTBench(DecodingModel):
    def __init__(self, parser_args):
        super().__init__(None, None, None, parser_args=parser_args)
        self.nodes_per_layer: int = parser_args.nodes_per_layer
        self.max_depth: int = parser_args.max_depth
        self.dataset_name = "MT-bench"
        # load relative resources
        self.load_data()
        self.seed_set = set()

        self.model_id = "llama-3.1"
        # if "Llama-2" in self.parser_args.draft_model and "Llama-2" in self.parser_args.target_model:
        #     self.model_id = "llama-2-chat"
        # elif "vicuna" in self.parser_args.draft_model and "vicuna" in self.parser_args.target_model:
        #     self.model_id = "vicuna"
        # elif "Llama-3.1" in self.parser_args.draft_model and "Llama-3.1" in self.parser_args.target_model:
        #     self.model_id = "llama-3.1"
        # else:
        #     raise NotImplementedError

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading MT-bench data...", 3)
        data = []
        with open(os.path.join(self.parser_args.data_path, "mt_bench.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        pass

    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def eval(self):
        # if self.parser_args.eval_mode == "two_model":
        out_path = os.path.join(self.parser_args.exp_name, f"{self.parser_args.eval_mode}_mt_bench.jsonl")
        out_f = open(out_path, "a")

        for question in tqdm.tqdm(self.data, total=len(self.data), disable=not self.is_target_model, ncols=50):

            choices = []
            # set random seed. Ensure each experiment runs with a unique random seed.
            for i in range(self.parser_args.num_samples_per_task):
                while self.seed in self.seed_set:
                    self.seed = random.randint(0, 1000000)
                seed_everything(self.seed)

                if self.model_id == "llama-3.1":
                    messages = [
                        {"role": "system",
                         "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    ]
                else:
                    conv = get_conversation_template(self.model_id)
                    if self.model_id == "llama-2-chat":
                        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                        conv.system_message = sys_p

                turns = []
                wall_time = []
                num_token = []
                for turn_idx in range(len(question["turns"])):
                    qs = question["turns"][turn_idx]

                    if self.model_id == "llama-3.1":
                        messages.append({
                            "role": "user",
                            "content": qs
                        })
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        input_ids = torch.tensor(self.tokenizer([prompt], add_special_tokens=False, ).input_ids).to(self.device)

                    else:
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt() + " "
                        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    if self.is_target_model:
                        output_ids = self.decoding_with_cache(input_ids, self.nodes_per_layer, self.max_depth)
                        # 结束后通知 drafter 结束
                        end_flag = torch.tensor(-1, device=self.model.device, dtype=torch.int)
                        dist.send(end_flag, dst=Config.DRAFTER_RANK)
                    if self.is_drafter:
                        self.draft(input_ids, self.nodes_per_layer, self.max_depth)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    dist.barrier()
                    if self.is_target_model:
                        output_text = self.tokenizer.decode(output_ids[0], spaces_between_special_tokens=False)

                        for special_token in self.tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output_text = output_text.replace(special_tok, "")
                            else:
                                output_text = output_text.replace(special_token, "")
                        output_text = output_text.strip()
                        if self.model_id == "llama-3.1":
                            messages.append({
                                "role": "assistant",
                                "content": output_text
                            })
                        else:
                            conv.messages[-1][-1] = output_text
                        turns.append(output_text)
                        wall_time.append(end_time - start_time)
                        num_token.append(output_ids.shape[1] - input_ids.shape[1])
                choices.append({"index": i, "wall_time": wall_time, "num_token": num_token, "turns": turns})

            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": self.model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            if self.is_target_model:
                out_f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
                out_f.flush()
        out_f.close()

        self.color_print(f"current eval mode: {self.parser_args.eval_mode}", 0)

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
    alg = EvalMTBench(parser_args)
    alg.eval()
