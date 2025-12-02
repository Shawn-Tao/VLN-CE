def merge_actions(action_list, max_length):
    if not action_list:
        return [], []
    
    # 分割连续的动作组
    groups = []
    current_value = action_list[0]
    current_start = 0
    for i in range(1, len(action_list)):
        if action_list[i] != current_value:
            groups.append((current_value, current_start, i-1))
            current_value = action_list[i]
            current_start = i
    groups.append((current_value, current_start, len(action_list)-1))
    
    merged_actions = []
    merged_indices = []
    
    # 处理每个组，分割为子组
    for value, start, end in groups:
        total_length = end - start + 1
        current_pos = start
        remaining = total_length
        
        while remaining > 0:
            current_segment_length = min(max_length, remaining)
            segment_end = current_pos + current_segment_length - 1
            # 合并后的动作字符串
            merged_actions.append(value * current_segment_length)
            # 对应的索引列表
            merged_indices.append(list(range(current_pos, segment_end + 1)))
            remaining -= current_segment_length
            current_pos = segment_end + 1
    
    return merged_actions, merged_indices

# 示例输入
actions = ['2', '2', '2', '2', '2', '2', '2', '2', '1', '1', '1', '1', '3', '1', '1', '3', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '1', '1', '3', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '1', '1', '1', '1', '2', '1', '1', '1', '1', '1', '1', '1', '1', '3', '3', '3', '3', '3', '3', '1', '1', '1', '3', '1', '0']
max_len = 3  # 可调整最长合成长度

merged_actions, merged_indices = merge_actions(actions, max_len)

print("合并后的动作列表:", merged_actions)
print("对应的索引列表:", merged_indices)