* dependency
```shell
pip3 install transformers==4.45.2 tqdm ipdb accelerate numpy shortuuid fschat fastchat
```
* specify python env 
```shell
export PYTHONPATH=$PYTHONPATH::/home/TreeDecoding/
```
* test env . parallel decoding
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-8B-Instruct --max_tokens 512 
```
* test env：single model decoding
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode single_model  --model_name Llama-3.1-8B-Instruct --max_tokens 512 
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode single_model  --model_name Llama-3.1-70B-Instruct --max_tokens 512 
```

* profile 70B model
```shell
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50 --communication_ratio 6
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_humaneval.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mgsm.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mt_bench.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_mbpp.py --eval_mode two_model --draft_model Llama-3.1-8B-Instruct  --target_model Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 50

```
* test autoregressive decoding
```shell
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 ParallelDecodingModel.py --eval_mode single_model  --model_name Llama-3.1-70B-Instruct --max_tokens 512 --nodes_per_layer 100
```
* cd benchmark
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct  --target_model Llama-3.1-8B-Instruct --max_tokens 512 
```
* profile autoregressive executation time
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode single_model  --model_name Llama-3.1-8B-Instruct --max_tokens 512
```
