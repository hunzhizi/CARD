import json
import time
from pathlib import Path
import numpy as np
import torch

from drafter_decoding.DecodingModel import DecodingModel
from drafter_decoding.util import *
import tqdm
import torch.distributed as dist


class MBPPEvaluator(DecodingModel):
    def __init__(self, parser_args):
        super().__init__(None, None, None, parser_args=parser_args)
        self.nodes_per_layer: int = parser_args.nodes_per_layer
        self.max_depth: int = parser_args.max_depth
        self.dataset_name = "MBPP"
        self.encode_special_token_flag = not (
                "Llama-3.2-1B-Instruct" in self.parser_args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.parser_args.target_model)

        self.seed_set = set()
        self.load_data()

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading MBPP data...", 3)
        data = []
        max_samples = 100  # 最大加载样本数
        with open(os.path.join(self.parser_args.data_path, "mbpp.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["text"])
                encode_special_token_flag = not (
                        "Llama-3.2-1B-Instruct" in self.parser_args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.parser_args.target_model)

                input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=encode_special_token_flag)
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                data.append(datum)
                if len(data) > max_samples:
                    break
        self.data = data


    def preprocess(self, input_text):
        text = input_text.strip()
        return text

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text) + len(
                self.tokenizer.bos_token) + 1:]  # tokenizer will add a '<s> ' at the beginning of the text.
        else:
            generation = output_text[len(input_text):]
        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                next_line = generation.index(stop_word)
                generation = generation[:next_line].strip()
        output_text = input_text + '\n    ' + generation
        output_text = output_text.replace("\t", "    ")

        return output_text

    @torch.no_grad()
    def eval(self):
        # if self.parser_args.eval_mode == "two_model":
        out_path = os.path.join(self.parser_args.exp_name, f"{self.parser_args.eval_mode}_{self.dataset_name}.jsonl")
        out_f = open(out_path, "a")
        for _ in range(self.parser_args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)

            for datum in tqdm.tqdm(self.data, total=len(self.data), disable=not self.is_target_model, ncols=50):
                input_ids = datum["input_ids"].to(self.device)
                torch.cuda.synchronize()
                if self.parser_args.eval_mode == "two_model":
                    if self.is_target_model:
                        start_time = time.time()
                        generate_ids = self.decoding_with_cache_sycn(input_ids, self.nodes_per_layer,
                                                                     self.max_depth)
                        end_time = time.time()
                    if self.is_drafter:
                        self.draft(input_ids, self.nodes_per_layer, self.max_depth)
                elif self.parser_args.eval_mode == "single_model":
                    start_time = time.time()
                    generate_ids = self.autoregressive_decoding(input_ids)
                    end_time = time.time()
                torch.cuda.synchronize()
                dist.barrier()
                if self.is_target_model:
                    output = self.postprocess(datum["input_text"], self.tokenizer.decode(generate_ids[0, :]))
                    out_f.write(json.dumps({"task_id": datum["task_id"], "time": end_time - start_time,
                                            "new_tokens": generate_ids.shape[1] - input_ids.shape[1],
                                            "completion": output}, ensure_ascii=False) + "\n")
                out_f.flush()

        out_f.close()

        self.color_print(f"current eval mode: {self.parser_args.eval_mode}", 0)


if __name__ == "__main__":
    parser_args = parse_arguments()
    alg = MBPPEvaluator(parser_args)
    alg.eval()