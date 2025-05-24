import time

from drafter_decoding.DecodingModel import DecodingModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from util import *
import torch.distributed as dist


class ParallelDecodingModel(DecodingModel):
    def __init__(self, parser_args):
        super().__init__(None, None, None,parser_args=parser_args )
        self.nodes_per_layer: int = parser_args.nodes_per_layer
        self.max_depth: int = parser_args.max_depth

    def eval(self):
        encode_special_token_flag = not (
                    "Llama-3.2-1B-Instruct" in self.parser_args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.parser_args.target_model)
        input_ids = self.tokenizer.encode(
            "Write an epic, multi-chapter story following a young thief named Lira who steals a cursed artifact from a forgotten temple, unwittingly awakening an ancient entity that threatens to destroy her war-torn homeland. Include diverse allies, dangerous rivals, and a richly imagined world—describe every scene in detail, develop characters deeply, and extend the plot as far as possible.",
            add_special_tokens=encode_special_token_flag)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        if self.parser_args.eval_mode == "two_model":
            if self.is_target_model:
                generate_ids = self.decoding_with_cache_sycn(input_ids, self.nodes_per_layer, self.max_depth)
                # 结束后通知 drafter 结束
            if self.is_drafter:
                self.draft(input_ids, self.nodes_per_layer, self.max_depth)
        elif self.parser_args.eval_mode == "single_model":
            start_time = time.time()
            generate_ids = self.autoregressive_decoding(input_ids)
            end_time = time.time()
            new_tokens = generate_ids.shape[1] - input_ids.shape[1]
            print(f" time is {end_time - start_time}, tokens/s is {new_tokens/(end_time - start_time)}")




if __name__ == '__main__':
    parser_args =parse_arguments()
    model = ParallelDecodingModel(parser_args)
    model.eval()




























