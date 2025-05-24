## llama3 1B & 3B  for test
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.2-3B-Instruct --max_tokens 512 --nodes_per_layer 20 --communication_ratio 2
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.2-3B-Instruct --max_tokens 512 --nodes_per_layer 20 --communication_ratio 2
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.2-3B-Instruct --max_tokens 512 --nodes_per_layer 20 --communication_ratio 2
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.2-3B-Instruct --max_tokens 512 --nodes_per_layer 20 --communication_ratio 2
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.2-3B-Instruct --max_tokens 512 --nodes_per_layer 20 --communication_ratio 2
#
## llama3 70B & 8B
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50 --communication_ratio 4
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50 --communication_ratio 4
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50 --communication_ratio 4
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50 --communication_ratio 4
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50 --communication_ratio 4
#
#
# llama3 70B & 1B
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100 --communication_ratio 7 --exp_name "llama3-CARD"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100 --communication_ratio 7 --exp_name "llama3-CARD"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100 --communication_ratio 7 --exp_name "llama3-CARD"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100 --communication_ratio 7 --exp_name "llama3-CARD"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100 --communication_ratio 7 --exp_name "llama3-CARD"

# llama2 70B & 7B
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5

# llama2 70B & 7B (temp=1)
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5 --temperature 1 --exp_name "llama2-temperature1"
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5 --temperature 1 --exp_name "llama2-temperature1"
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-2-7b-chat-hf  --target_model Llama-2-70b-chat-hf --max_tokens 512 --nodes_per_layer 50 --communication_ratio 5 --temperature 1 --exp_name "llama2-temperature1"

#
## llama3 70b autoregressive decoding bsline
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 200
#
#
## llama2 70b autoregressive decoding bsline
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 200
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 200
