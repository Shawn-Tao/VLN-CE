
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
import orjson 
import numpy as np
import re

# import 当前目录下的softmax_sample文件夹，使用os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'softmax_sample')))
from datetime import datetime

from softmax_sample.reject_generate import ACTION_CLASS,sample_reject_action_softmax,STR_TO_ACTIONS,ACTIONS_TO_STR


STR_MAT_TO_CMD = {
    "move forward 0.25 meters": "move_forward 0.25",
    "move forward 0.5 meters": "move_forward 0.5",
    "move forward 0.75 meters": "move_forward 0.75",
    "turn left 15 degrees": "turn_left 15",
    "turn left 30 degrees": "turn_left 30",
    "turn left 45 degrees": "turn_left 45",
    "turn right 15 degrees": "turn_right 15",
    "turn right 30 degrees": "turn_right 30",
    "turn right 45 degrees": "turn_right 45",
    "stop":              "stop"
}
def list_first_level(path):
    return [str(child.absolute()) for child in Path(path).iterdir()]

def list_first_level(path):
    contents = []
    with os.scandir(path) as entries:
        for entry in entries:
            contents.append(entry.path)  # 直接获取条目的完整路径
    return contents
    
def load_yamls_from_dir(config_dir: str):
    config = OmegaConf.create()  # 创建空配置
    config_dir = Path(config_dir)
    
    # 遍历目录下的所有 YAML 文件
    for yaml_file in config_dir.glob("*.yaml"):
        cfg = OmegaConf.load(yaml_file)
        config = OmegaConf.merge(config, cfg)  # 合并配置
    
    return config

def deep_merge(source: dict, overrides: dict) -> dict:
    """递归合并嵌套字典"""
    merged = source.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

# 合并策略
def merge_separately(*datasets):
    # 合并messages（保留所有记录）
    merged_messages = []
    # 合并images（去重后按数字升序排列）
    merged_images = set()  

    for data in datasets:
        merged_messages.extend(data["messages"])
        merged_images.update(data["images"])

    # 将图像ID转换为整数排序后再转回字符串
    # sorted_images = sorted(map(int, merged_images))
    return {
        "messages": merged_messages,
        "images": [str(img_id) for img_id in sorted_images]
    }

    
SYSTEM_PROMPT_FULL = """You are a navigation agent operating in a 3D indoor environment.

You control the agent using discrete actions that correspond to physical movements.

Action physical semantics:
- Forward actions move the agent straight ahead by a fixed distance measured in meters.
- Turning actions rotate the agent in place by a fixed angle measured in degrees.

Action definitions:
{action_definitions}

When selecting actions, consider:
- Smaller forward distances for fine position adjustment.
- Larger forward distances for approaching distant targets.
- Smaller rotation angles for fine heading correction.
- Larger rotation angles for major direction changes.

Action selection rules:
- Output exactly ONE action.
- The output must exactly match one of the action names above.
- Do not output explanations or reasoning.

If the task is complete, output:
stop"""


SYSTEM_PROMPT_MINIMAL = """You are a navigation agent.

Output exactly ONE action from the predefined action set.
Do not output explanations."""


action_definitions = "\n".join([
    "- move_forward 0.25: move forward by approximately 0.25 meters.",
    "- move_forward 0.5: move forward by approximately 0.50 meters.",
    "- move_forward 0.75: move forward by approximately 0.75 meters.",
    "- turn_left 15: rotate left by approximately 15 degrees.",
    "- turn_left 30: rotate left by approximately 30 degrees.",
    "- turn_left 45: rotate left by approximately 45 degrees.",
    "- turn_right 15: rotate right by approximately 15 degrees.",
    "- turn_right 30: rotate right by approximately 30 degrees.",
    "- turn_right 45: rotate right by approximately 45 degrees.",
    "- stop: no movement."
])

class Json_Templater:
    def __init__(self):
        self.config = load_yamls_from_dir("generate_json_dpo")
        date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_file_save_path = f'{date_prefix}_{self.config.qwen_r2r_dataset_cfg.json_file_save_path}'
        self.dataset_file_path = self.config.qwen_r2r_dataset_cfg.dataset_file_path
        
        # self.template_header_str = f"You are a robot programmed for navigation tasks. You have been given "
        # self.temlpate_history_str = f"a series of historical observations:<history_image> and "
        # self.template_current_str = f"the current observation: <image>. "
        # self.template_last_action_str = f"Your last action is: \"<action>\". "
        
        # self.template_task_str = f"Your assigned task is: \"<task>\". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or stop if the task is complete."
        self.template_system_str = SYSTEM_PROMPT_FULL.format(action_definitions=action_definitions)
        
        self.template_user_history_str = f"Historical observations: <history_image>\n"
        self.template_user_current_str = f"Current observation: <image>\n"
        self.template_user_action_str = f"Last action: \"<action>\"\n"
        self.template_user_task_str = f"Task: \"<task>\"\n"        
        
        
        self.max_historical_image_num = self.config.qwen_r2r_dataset_cfg.max_historical_image_num
        self.max_action_sequence_length = self.config.qwen_r2r_dataset_cfg.max_action_sequence_length
        
        self.path_prefix = self.config.qwen_r2r_dataset_cfg.path_prefix
        self.reduced_image_path_prefix = self.config.qwen_r2r_dataset_cfg.reduced_image_path_prefix
        self.if_reduce_history_image = self.config.qwen_r2r_dataset_cfg.if_reduce_history_image
        
        self.uniform_sample_history = self.config.qwen_r2r_dataset_cfg.uniform_sample_history

    def create_json_template(self):
        # Create the JSON template
        json_data = {
            "conversions": [],
            "chosen":{},
            "reject":{},
            "image": []
        }
        return json_data
    
    def merge_actions(self, action_list, max_length):
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

    # 线性等距采样
    def uniform_sample(self, arr, n):
        indices = np.linspace(0, len(arr)-1, n, dtype=int)  # 生成均匀分布的索引
        return [arr[i] for i in indices]

    def fill_in_json_template(self):
        folder_list = list_first_level(self.dataset_file_path)
        r2r_data_set_json = []
        folder_count = 0
        start_time = time.time()
        current_time = time.time()
        end_time = time.time()
        
        total_stop_count = 0
        
        for folder in folder_list:
            folder_count += 1
            if(folder_count % 100 == 0):
                end_time = time.time()
                duration = end_time - current_time
                totol_duration = end_time - start_time
                current_time = time.time()
                print(f"{folder_count} / {len(folder_list)} , {duration} , {totol_duration}")
            
            episode_id = folder.split("/")[-1]
            
            instruction_str = ""
            action_str = ""
            priv_info = ""
            
            instruction_save_path  = os.path.join(folder, f"{episode_id}_instruction.txt")
            with open(instruction_save_path, "r") as f:
                instruction_str = f.read()
            
            action_save_path = os.path.join(folder, f"{episode_id}_actions.txt")
            with open(action_save_path, "r") as f:
                action_str = f.read()
            action_list = action_str.split(", ")
            # print(action_list)
            
            json_info_save_path = os.path.join(folder, f"{episode_id}_info.json")
            with open(json_info_save_path, "r", encoding="utf-8") as f:
                priv_info = json.load(f)
            
            merged_actions, merged_indices = self.merge_actions(action_list, 3)
            # print(merged_actions)
            
            if self.if_reduce_history_image:
                # 如果需要缩小历史图像的分辨率，则将图片路径前缀改为reduced_image_path_prefix
                self.prefix = self.reduced_image_path_prefix
            else:
                # 如果不需要缩小历史图像的分辨率，则将图片路径前缀改为path_prefix
                self.prefix = self.path_prefix
            
            # 每个动作生成一条数据
            for step in range(len(merged_actions)):
                # 单个动作生成一条数据
                # json_data= self.create_json_template()
                action_str = self.action_to_str(merged_actions[step])
                reject_action_str = "stop"
                # print(action_str)

                # json_assistant = {
                #     "conversations": [

                #     ]
                # }
                
                header_str = ""
                history_str = ""
                current_str = ""
                task_str = ""
                history_action_str = ""
                
                # 不用绝对路径了，改成相对路径，因为docker里面找路径和主机不一样
                image_list = []
                
                header_str = self.template_system_str
                current_str = self.template_user_current_str
                task_str = self.template_user_task_str.replace("<task>", instruction_str)
                # task_str = f'{self.template_task_str}'.replace("<task>", instruction_str)
                
                if(step == 0):
                    image_list.append(f"{self.path_prefix}/{episode_id}/{episode_id}_{merged_indices[step][0]}.jpg")
                else:
                    historical_image_str = ""
                    if(step < self.max_historical_image_num):    
                        for i in range(step):
                            historical_image_str += f" <image>"
                        history_str = self.template_user_history_str.replace("<history_image>",historical_image_str)
                        for i in range(step):
                            image_list.append(f"{self.prefix}/{episode_id}/{episode_id}_{merged_indices[step-i-1][0]}.jpg")
                        image_list.reverse()  # 这里我们是倒着放的，所以需要反转
                        image_list.append(f"{self.path_prefix}/{episode_id}/{episode_id}_{merged_indices[step][0]}.jpg")
                    else:
                        for i in range(self.max_historical_image_num):
                            historical_image_str += f" <image>"
                        history_str = self.template_user_history_str.replace("<history_image>",historical_image_str)
                        
                        if self.uniform_sample_history:
                            start_to_current_indices = merged_indices[:step+1]
                            start_to_current_indices = self.uniform_sample(start_to_current_indices, self.max_historical_image_num+1)
                            for i in range(self.max_historical_image_num+1):
                                image_list.append(f"{self.prefix}/{episode_id}/{episode_id}_{start_to_current_indices[i][0]}.jpg")
                                
                        else:
                            for i in range(self.max_historical_image_num):
                                image_list.append(f"{self.prefix}/{episode_id}/{episode_id}_{merged_indices[step-i-1][0]}.jpg")
                            # 这里我们是倒着放的，所以需要反转
                            image_list.reverse()  
                            image_list.append(f"{self.path_prefix}/{episode_id}/{episode_id}_{merged_indices[step][0]}.jpg")
                
                # 截取先前的动作序列
                action_squence = []
                if step > 0:
                    if step < self.max_action_sequence_length:
                        action_squence = merged_actions[:step]
                    else:
                        action_squence = merged_actions[step - self.max_action_sequence_length:step]
                if len(action_squence) > 0:
                    # 将动作序号转换成字符串
                    action_squence_str = self.action_to_str(action_squence[-1])
                    action_squence_str = STR_MAT_TO_CMD[action_squence_str]
                    history_action_str = self.template_user_action_str.replace("<action>", action_squence_str)
                    
                #! 修改思路，当 distance_to_goal 小于一定值，将action_str修改为 "stop"，从而增加stop数据的数量。目前在 33w条数据中，stop数据仅有1w条左右，占比过少。
                #! 再次修改，生成更加丰富的偏好数据集
                tau = 3
                # if(priv_info["distance_to_goal"][merged_indices[step][0]] < 1.6):
                if(priv_info["distance_to_goal"][merged_indices[step][0]] < 1.1):
                    # print("change to stop")
                    if(action_str != "stop"):
                        action_str = "stop"
                    reject_action, _, _ = sample_reject_action_softmax(action_str, tau=tau)
                    reject_action_str = ACTIONS_TO_STR[reject_action]
                else:
                    reject_action, _, _ = sample_reject_action_softmax(STR_TO_ACTIONS[action_str], tau=tau)
                    reject_action_str = ACTIONS_TO_STR[reject_action]

                # print("chosen:", action_str, "  reject:", reject_action_str)
                        
                    
                # json_data = {
                #     # "id": total_data_count,
                #     "conversations": [
                #         {"from": "human", "value": f"{header_str}{history_str}{current_str}{history_action_str}{task_str}"},
                #         {"from": "gpt", "value": f"{action_str}"},
                #     ],
                #     "image": image_list,
                #     "goal_position":priv_info["goal_position"],
                #     "distance_to_goal":priv_info["distance_to_goal"][merged_indices[step][0]],
                #     "agent_position":priv_info["current_gps_pos"][merged_indices[step][0]],
                #     "agent_heading":priv_info["current_heading"][merged_indices[step][0]]
                # }
                
                action_str = STR_MAT_TO_CMD[action_str]
                reject_action_str = STR_MAT_TO_CMD[reject_action_str]
                
                
                json_data ={
                    "messages": [
                        {
                            "role": "system", 
                            "content": f"{header_str}"
                        },
                        {
                            "role": "user", 
                            "content": f"{history_str}{current_str}{history_action_str}{task_str}"
                        },
                    ],
                    "chosen":{"role": "assistant", "content": f"{action_str}"},
                    "rejected":{"role": "assistant", "content": f"{reject_action_str}"},
                    "images": image_list,
                }
                
                # json_data = {
                #     # "id": total_data_count,
                #     "conversations": [
                #         {
                #             "from": "system", 
                #             "value": f"{header_str}"
                #         },
                #         {
                #             "from": "human", 
                #             "value": f"{history_str}{current_str}{history_action_str}{task_str}"
                #         },
                #     ],
                #     "chosen":{"from": "gpt", "value": f"{action_str}"},
                #     "rejected":{"from": "gpt", "value": f"{   reject_action_str}"},
                #     "images": image_list,
                # }
                
                # json_data = {
                #     # "id": total_data_count,
                #     # "conversations": [
                #     #     {"from": "human", "value": f"{header_str}{history_str}{current_str}{history_action_str}{task_str}"},
                #     # ],
                #     "system": f"{header_str}",
                #     "instruction": f"{history_str}{current_str}{history_action_str}{task_str}",
                #     "chosen":f"{action_str}",
                #     "rejected":f"{reject_action_str}",
                #     "images": image_list,
                # }

                if(action_str == "stop"):
                    total_stop_count += 1
                
                    # print(f"distance_to_goal: {priv_info['distance_to_goal'][merged_indices[step][0]]}, action_str: {action_str}")
                
                # total_data_count +=1
                
                # 去掉json_data中的换行，以便写入jsonl文件中
                
                
                # 将json_data 写入 总的dict
                r2r_data_set_json.append(json_data)
        
        print("***************************************total_stop_count:",total_stop_count)
                
        # save json_data using orjson
        
        # with open(f"{folder}/{episode_id}.json", "w") as f:
        with open(f"{self.json_file_save_path}", "wb") as f:
            for item in r2r_data_set_json:
                f.write(orjson.dumps(item) + b"\n")
                # json_bytes = orjson.dumps(item, f, option=orjson.OPT_INDENT_2)
                # f.write(json_bytes)       
               
        # exit()
    def action_to_str(self, action):
        # Convert action to string
        if action == "0":
            return "stop"
        elif action[0] == "1":
            return f"move forward {0.25*len(action)} meters"
        elif action[0] == "2":
            return f"turn left {15*len(action)} degrees"
        elif action[0] == "3":
            return f"turn right {15*len(action)} degrees"
        else:
            return action
        
    
if __name__ == "__main__":
    json_templater = Json_Templater()
    json_templater.fill_in_json_template()