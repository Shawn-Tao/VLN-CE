import json

input_file = "20251228_205110_r2r_orpo_train_424_240_qwen3vl_1p1_tau3.jsonl"
output_file = "20251228_205110_r2r_orpo_train_424_240_qwen3vl_1p1_tau3.json"

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 忽略空行
            data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_file}")
