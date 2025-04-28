import torch.nn as nn
import torch
from fastchat.data.split_long_conversation import max_length
from numpy.ma.core import divide

from drafter_decoding.KVCacheModel import KVCacheModel
from drafter_decoding.Tree import Tree
from drafter_decoding.kv_cache import initialize_past_key_values
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import random


base_model_path = "/mnt/data/zhouShaoRepo/model/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
your_message="tell me a story about little bear."
input_ids=tokenizer([your_message]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

model = KVCacheModel(model,temperature=0,init_input_len=input_ids.shape[1])
# input_ids = model.generate(input_ids)

cache = torch.tensor([128000,  73457,    757,    264,   3446,    922,   2697,  11984,     13,
          2697,  11984,    374,    264,   2678,     11,  68661,  11984,    889,
          6439,    304,    264,  43535,   2697,  26457,    304,    279,  13952,
            13,    568,  16180,    311,   1514,   4994,    323,  13488,    279,
         33633,     11,    719,    568,    596,   1101,   1633,  22999,    323,
          3629,   5334,   1139,  12544,    627,   4054,  40798,  13658,     11,
         15013,  24941,    574,    704,   5737,    304,    279,  13952,     11,
         43931,   1306,  81776,    323,  58387,    287,   8545,  89770,     13,
          1283,   3782,   4028,    264,   2678,  33850,    323,    304,    279,
          4219,    315,    433,     11,    568,   5602,    264,    387,   2701,
           535,  21363,    505,    264,   5021,   9046,     13,  15013,  24941,
           596,  41328,   2751,    279,   1888,    315,   1461,     11,    323,
           568,   6773,    311,  19874,    279,  66607,    627,   1548,  25735,
           279,  66607,  14297,     11,    539,  19762,    311,  44030,    279,
         40558,   3201,     13,   2030,    439,    568,   2751,  12401,     11,
           568,  33484,  33085,   2403,    279,  66607,    449,    813,  77938,
            11,  14718,    279,  40558,    311,   3719,    945,  33337,     13,
          2435,  31527,    291,  87625,    323,  32122,    704,    315,    279,
         66607,     11,  43931,   1306,  15013,  24941,    627,  39203,  24941,
         10837,    439,   5043,    439,    813,   2697,  14535,   1436,   6920,
          1461,     11,    719,    279,  40558,   1051,   4106,    389,    813,
         34460,     13,   1283,  57067,    291,   1990,    279,  12690,     11,
           813,   4851,  22019,    449,   8850,     13,   4702,    994,    568,
          3463,    568,    574,   2133,    311,    636,  10791,     11,    568,
         27569,    264,  14397,    813,   6691,   1047,  15972,   1461,     13,
          1283,   6288,  37085,    291,   4920,    264,   3544,   7091,    323,
          5762,    813,  11745,     11,   8748,    369,    279,  40558,    311,
          1522,    555,    627,   4599,    279,  13962,    574,   2867,     11,
         15013,  24941,  22763,    505,   4920,    279,   7091,     11,  26346,
           287,    323,  66900,     13,   1283,   7111,    709,    520,    279,
         66607,    323,  15393,    430,    568,   1047,   1903,    264,   2466,
         16930,     13,   1283,   3287,    956,   1390,    311,    636,    357,
          2234,    555,    279,  40558,   1578,     11,    779,    568,   6773,
           311,   5387,    279,  66607,   7636,    505,   1457,    389,    627,
          2170,    568,   1903,    813,   1648,   1203,    311,    813,  26457,
            11,  15013,  24941,   7846,    956,   1520,    719,   1781,    922,
          1268,   3345,    568,   1047,   2586,    311,   3794,   1139,  12544,
            13,   1283,    574,  26259,    369,    813,   4062,   7422],
       device='cuda:0')
seq_len = input_ids.shape[1]
input_ids = torch.cat([input_ids, cache[seq_len: seq_len + 3].unsqueeze(0)],dim=-1)
output_ids = input_ids
# decode phase
while True:
    # 1. query cache
    # best_candidates, picked_index = cache_manager.query_cache()
    # 2. decode with cache
    # input_ids = torch.cat([input_ids,best_candidates], dim=-1)
    input_ids = model.generate(input_ids)
    print(tokenizer.decode(input_ids))
    randint = random.randint(1,3)
    if randint > 0:
        seq_len = output_ids.shape[1]
        if seq_len >= cache.shape[0]:
            break
        output_ids = torch.cat([output_ids, cache[seq_len: seq_len + randint].unsqueeze(0)],dim=-1)
        print(tokenizer.decode(output_ids[0]))
        input_ids = cache[seq_len: seq_len + randint].unsqueeze(0)


    # print(tokenizer.decode(input_ids[0]))
    # print(input_ids[0])

    # 3. verify
    # 4. isend message to drafter
    #   4.1 receive (could find a path in cache)
    #   4.2 reject  (couldnot find a path in cache)

    # 5.  rollback kv_cache
    # 6. update tree

