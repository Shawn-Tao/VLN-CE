import json
import random

def load_and_shuffle_jsonl(input_path, output_path=None, seed=None):
    """
    读取 jsonl 文件并打乱顺序，可选保存为新的 jsonl 文件
    Args:
        input_path (str): 输入 jsonl 文件路径
        output_path (str, optional): 输出 jsonl 文件路径，不指定则不保存
        seed (int, optional): 随机种子，保证可复现
    Returns:
        list: 打乱后的数据列表
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    # 如果指定了输出路径，保存打乱后的数据
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return data


# 使用示例
if __name__ == "__main__":
    input_file = "20251201_r2r_orpo_train_424_240_more_stop_sampled_0dot25.jsonl"
    output_file = "20251201_r2r_orpo_train_424_240_more_stop_sampled_0dot25_shuffle.jsonl"
    shuffled_data = load_and_shuffle_jsonl(input_file, output_file, seed=114514)
    print(f"共加载 {len(shuffled_data)} 条数据，已保存至 {output_file}")
    print(shuffled_data[:5])
    