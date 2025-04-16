from opt_classic.model import SPModel
from fastchat.model import get_conversation_template
import torch

base_model_path = "/mnt/data/zhouShaoRepo/EAGLE/eagle/model_weight/base_model"
# base_model_path = "/mnt/data/zhouShaoRepo/model/Llama-3.1-8B-Instruct"
draft_model_path = "/mnt/data/zhouShaoRepo/model/llama-68m"

model = SPModel.from_pretrained(
    base_model_path=base_model_path,
    draft_model_path=draft_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

your_message="Hello, tell me a story about a man who lost his way in the forest and found a treasure."

input_ids=model.tokenizer([your_message]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.spgenerate(input_ids,temperature=0,max_new_tokens=1024,nodes=50,threshold=0.7,max_depth=10)
output=model.tokenizer.decode(output_ids[0])

print(output)
