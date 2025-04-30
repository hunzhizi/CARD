import time

from transformers import AutoTokenizer, AutoModelForCausalLM

from drafter_decoding.DecodingModel import DecodingModel
import torch

from drafter_decoding.KVCacheModel import KVCacheModel
from drafter_decoding.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM

# base_model_path = "/mnt/data/zhouShaoRepo/EAGLE/eagle/model_weight/base_model"
# base_model_path = "/mnt/data/zhouShaoRepo/model/Llama-3.1-8B-Instruct"
# base_model_path = "/hy-tmp/Llama-3.1-8B-Instruct"
base_model_path = "/hy-tmp/Llama-3.2-1B-Instruct"
draft_model_path = "/mnt/data/zhouShaoRepo/model/llama-68m"



def test_draft_single_card():
    model = DecodingModel.from_pretrained(
        draft_model_path=base_model_path,
        main_device='cuda',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()

    your_message = "Hello, tell me a story about a man who lost his way in the forest and found a treasure."
    # your_message="tell me a story about Little bear."
    input_ids = model.tokenizer([your_message]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    model.device = 'cuda'
    # output_ids=model.draft_single_card_test(input_ids,nodes_per_layer=20)
    output_ids=model.autoregressive_decoding(input_ids)
    output=model.tokenizer.decode(output_ids[0])

    print(output)

def test_autoregressive_decoding():
    # output_ids=model.autoregressive_decoding(input_ids)
    # # output_ids=model.autogressive_decoding(input_ids)
    # output=model.tokenizer.decode(output_ids[0])
    #
    # print(output)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Set padding side to left to maintain autoregressive property
    tokenizer.padding_side = "left"

    # Add special tokens if needed for your model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input for model
    your_message = "tell me a 20 words story"
    inputs = tokenizer(your_message, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    # Store the original input to preserve it during generation
    original_input_length = input_ids.shape[1]

    # Set generation parameters
    max_new_tokens = 600  # Maximum number of tokens to generate
    temperature = 1.0  # Control randomness (lower = more deterministic)

    # Generation loop
    output_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        # Get only the most recent context to avoid excessive memory usage
        # Optional, especially useful for longer generations
        if output_ids.shape[1] > 1024:  # Example context window size
            context_ids = output_ids[:, -1024:]
        else:
            context_ids = output_ids

        with torch.no_grad():  # No need to track gradients during inference
            outputs = model(context_ids)

        # Get logits for the last token
        next_token_logits = outputs.logits[:, -1, :]

        # Optional: Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Get the most likely next token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # Append the new token to the sequence
        output_ids = torch.cat([output_ids, next_token.transpose(0, 1)], dim=1)

        # Print the current output (optional - can be removed for performance)
        current_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated so far: {current_text}")

        # Stop if we generate an EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Get the final generated text
    generated_text = tokenizer.decode(output_ids[0][original_input_length:], skip_special_tokens=True)
    print(f"\nFinal generated text:\n{generated_text}")
    sum_tokens = output_ids[0][original_input_length:].shape[0]
    return output_ids, generated_text,sum_tokens


if __name__ == '__main__':
    start = time.perf_counter()
    output_ids, generated_text,sum_tokens = test_autoregressive_decoding()
    gap = time.perf_counter() - start
    print(f"执行时间 is {gap},生成 tokens{sum_tokens}, tokens/s = { sum_tokens/gap}")
    # test_draft_single_card()

