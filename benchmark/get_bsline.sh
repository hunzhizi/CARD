# llama2 70b autoregressive decoding bsline
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 100
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 100
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 100
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 100
#CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode single_model --model_name Llama-2-70b-chat-hf --max_tokens 100

# llama3 70b autoregressive decoding bsline
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 100 --exp_name "llama3-baseln"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 100 --exp_name "llama3-baseln"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 100 --exp_name "llama3-baseln"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 100 --exp_name "llama3-baseln"
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode single_model --model_name Llama-3.1-70B-Instruct --max_tokens 100 --exp_name "llama3-baseln"

