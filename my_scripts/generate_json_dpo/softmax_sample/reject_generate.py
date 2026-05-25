import numpy as np

ACTIONS_TO_STR = {
    "turn_left_45":   "turn left 45 degrees",
    "turn_left_30":   "turn left 30 degrees",
    "turn_left_15":   "turn left 15 degrees",

    "forward_0.25":   "move forward 0.25 meters",
    "forward_0.5":    "move forward 0.5 meters",
    "forward_0.75":   "move forward 0.75 meters",

    "turn_right_15":  "turn right 15 degrees",
    "turn_right_30":  "turn right 30 degrees",
    "turn_right_45":  "turn right 45 degrees",

    "stop":           "stop",
}

STR_TO_ACTIONS = {
    "turn left 45 degrees": "turn_left_45",
    "turn left 30 degrees": "turn_left_30",
    "turn left 15 degrees": "turn_left_15",
    
    "move forward 0.25 meters": "forward_0.25", 
    "move forward 0.5 meters": "forward_0.5",
    "move forward 0.75 meters": "forward_0.75",
    
    "turn right 15 degrees": "turn_right_15",
    "turn right 30 degrees": "turn_right_30",
    "turn right 45 degrees": "turn_right_45",

    "stop":           "stop",
}

# 二维动作映射
ACTIONS = {
    "turn_left_45":   np.array([-45,  0]),
    "turn_left_30":   np.array([-30,  0]),
    "turn_left_15":   np.array([-15,  0]),

    "forward_0.25":   np.array([  0,  15]),
    "forward_0.5":    np.array([  0,  30]),
    "forward_0.75":   np.array([  0,  45]),

    "turn_right_15":  np.array([ 15,  0]),
    "turn_right_30":  np.array([ 30,  0]),
    "turn_right_45":  np.array([ 45,  0]),

    "stop":           np.array([  0,   0]),
}

# 动作分类
ACTION_CLASS = {
    "turn_left_45": "turn_left",
    "turn_left_30": "turn_left",
    "turn_left_15": "turn_left",

    "forward_0.25": "forward",
    "forward_0.5":  "forward",
    "forward_0.75": "forward",

    "turn_right_15": "turn_right",
    "turn_right_30": "turn_right",
    "turn_right_45": "turn_right",

    "stop": "stop"
}

# Motion embedding: unit-commensurate normalized vectors [theta/15deg, d/0.25m]
MOTION_EMBEDDING = {
    "turn_left_45":   np.array([-3,  0]),
    "turn_left_30":   np.array([-2,  0]),
    "turn_left_15":   np.array([-1,  0]),
    "forward_0.25":   np.array([ 0,  1]),
    "forward_0.5":    np.array([ 0,  2]),
    "forward_0.75":   np.array([ 0,  3]),
    "turn_right_15":  np.array([ 1,  0]),
    "turn_right_30":  np.array([ 2,  0]),
    "turn_right_45":  np.array([ 3,  0]),
    "stop":           np.array([ 0,  0]),
}

# Precompute raw dissim range across all cross-class pairs
_DISSIM_VALUES = []
for _a, _va in MOTION_EMBEDDING.items():
    for _b, _vb in MOTION_EMBEDDING.items():
        if _a == _b or ACTION_CLASS[_a] == ACTION_CLASS[_b]:
            continue
        _DISSIM_VALUES.append(np.linalg.norm(_va - _vb))
DISSIM_RAW_MIN = min(_DISSIM_VALUES)
DISSIM_RAW_MAX = max(_DISSIM_VALUES)


def compute_dissim(chosen_action: str, rejected_action: str) -> float:
    """Compute normalized Euclidean distance in the motion embedding space."""
    return float(np.linalg.norm(
        MOTION_EMBEDDING[chosen_action] - MOTION_EMBEDDING[rejected_action]
    ))


def scale_dissim(raw_dissim: float, min_val: float = 0.5, max_val: float = 1.5) -> float:
    """Scale raw dissim to [min_val, max_val] range linearly."""
    if DISSIM_RAW_MAX == DISSIM_RAW_MIN:
        return min_val
    return min_val + (raw_dissim - DISSIM_RAW_MIN) / (DISSIM_RAW_MAX - DISSIM_RAW_MIN) * (max_val - min_val)


def softmax(x):
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))

def sample_reject_action_softmax(chosen_action, tau=20.0):
    """
    chosen_action: str, 正确动作
    tau: float, softmax 的温度（越小越倾向选择远动作）
    """
    chosen_vec = ACTIONS[chosen_action]
    chosen_class = ACTION_CLASS[chosen_action]
    
    candidates = []
    distances = []

    for act, vec in ACTIONS.items():
        if act == chosen_action:
            continue
        
        if ACTION_CLASS[act] == chosen_class:
            # 同类动作直接忽略
            continue
        
        dist = np.linalg.norm(vec - chosen_vec)
        candidates.append(act)
        distances.append(dist)

    distances = np.array(distances)
    
    # Softmax over distances
    prob = softmax(distances / tau)

    # 按概率采样拒绝动作
    reject_action = np.random.choice(candidates, p=prob)

    return reject_action, prob, candidates

def generate_preference_dataset_softmax_from_sequence(action_sequence, tau=15):
    dataset = []
    for act in action_sequence:
        reject, _, _ = sample_reject_action_softmax(act, tau=tau)
        dataset.append({"chosen": act, "rejected": reject})
    
    print(dataset)
    return dataset


# def generate_preference_dataset_softmax(action, tau=15):
#     sample_reject_action_softmax(action, tau=tau)
    
#     dataset = []
    
#     for act in action_sequence:
#         reject, _, _ = sample_reject_action_softmax(act, tau=tau)
#         dataset.append({"chosen": act, "rejected": reject})
    
#     print(dataset)
#     return dataset

# action_sequence = ["forward_0.5"]
# generate_preference_dataset_softmax_from_sequence(action_sequence, tau=15)

if __name__ == "__main__":
    test_tau = 6

    reject, prob, candidates = sample_reject_action_softmax("turn_left_45", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("turn_left_30", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")    
    reject, prob, candidates = sample_reject_action_softmax("turn_left_15", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("turn_left_45", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("turn_left_30", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("turn_left_15", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("forward_0.25", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("forward_0.5", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("forward_0.75", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")
    reject, prob, candidates = sample_reject_action_softmax("stop", tau=test_tau)
    print("Reject:", reject)
    for c, p in zip(candidates, prob):
        print(f"{c:15s}  prob={p:.3f}")
    print("-----")

