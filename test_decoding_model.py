from transformers import AutoTokenizer

from drafter_decoding.DecodingModel import DecodingModel
import torch
from drafter_decoding.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM

# base_model_path = "/mnt/data/zhouShaoRepo/EAGLE/eagle/model_weight/base_model"
base_model_path = "/mnt/data/zhouShaoRepo/model/Llama-3.1-8B-Instruct"
draft_model_path = "/mnt/data/zhouShaoRepo/model/llama-68m"

model = DecodingModel.from_pretrained(
    draft_model_path=base_model_path,
    main_device='cuda',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

your_message="Hello, tell me a story about a man who lost his way in the forest and found a treasure."

input_ids=model.tokenizer([your_message]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.draft(input_ids,nodes_per_layer=20)
# output_ids=model.autogressive_decoding(input_ids)
output=model.tokenizer.decode(output_ids[0])

print(output)



