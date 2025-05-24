#!/bin/bash

# 基础命令部分（不含环境变量）
BASE_CMD="torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=12345 eval_gsm8k.py --eval_mode two_model --draft_model Llama-3.2-1B-Instruct --target_model Llama-3.1-70B-Instruct --max_tokens 512 --communication_ratio 7 --exp_name llama3_expore_cache_size"

# 初始化节点数
nodes=5

# 循环执行命令
while [ $nodes -le 205 ]; do
    echo "运行 nodes_per_layer = $nodes"
    # 正确设置环境变量并执行命令
    CUDA_VISIBLE_DEVICES=0,1,2 $BASE_CMD --nodes_per_layer $nodes

    # 增加节点数
    nodes=$((nodes + 20))
done

echo "所有nodes_per_layer参数测试完成！"