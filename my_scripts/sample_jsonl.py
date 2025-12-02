import json
import random

def sample_jsonl(in_file, out_file, ratio=0.1, seed=42):
    random.seed(seed)
    with open(in_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if random.random() < ratio:
                fout.write(line)

# 示例：抽取 10% 的数据
sample_jsonl("20251201_r2r_orpo_train_424_240_more_stop.jsonl", "20251201_r2r_orpo_train_424_240_more_stop_sampled_0dot25.jsonl", ratio=0.25)