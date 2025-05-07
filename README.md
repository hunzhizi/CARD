* 环境
pip3 install transformers==4.45.2 tqdm ipdb accelerate numpy shortuuid fschat fastchat
* 服务器运行需要指定python搜索环境 
export PYTHONPATH=$PYTHONPATH::/home/TreeDecoding/
* 测试环境：两个模型进行并行推理测试
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-8B-Instruct --max_tokens 512 
* 测试环境：单个模型进行推理
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode single_model  --model_name Llama-3.1-8B-Instruct --max_tokens 512 

* 测试70B 模型
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
* 测试autoregressive decoding
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode single_model  --model_name Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100
* 执行benchmark cd benchmark
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-8B-Instruct --max_tokens 512 
* 测试 autoregressive 的执行时间
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 benchmark/eval_gsm8k.py --eval_mode single_model  --model_name Llama-3.1-8B-Instruct --max_tokens 512