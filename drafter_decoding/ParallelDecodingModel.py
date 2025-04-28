from drafter_decoding.DecodingModel import DecodingModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from util import *


class ParallelDecodingModel(DecodingModel):
    def __init__(self, parser_args):
        super().__init__(None, None, None,parser_args=parser_args )
        self.nodes_per_layer: int = parser_args.nodes_per_layer
        self.max_depth: int = parser_args.max_depth

    def eval(self):
        if self.parser_args.eval_mode == "two_model":
            encode_special_token_flag = not (
                        "Llama-3.2-1B-Instruct" in self.parser_args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.parser_args.target_model)
            input_ids = self.tokenizer.encode(
                "tell me a story about little bear.",
                add_special_tokens=encode_special_token_flag)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            if self.is_target_model:
                self.decoding_with_cache(input_ids,self.nodes_per_layer, self.max_depth)
            if self.is_drafter:
                self.draft(input_ids, self.nodes_per_layer, self.max_depth)




if __name__ == '__main__':
    parser_args =parse_arguments()
    model = ParallelDecodingModel(parser_args)
    model.eval()




























