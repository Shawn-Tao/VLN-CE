
# 最终生成的 json 训练数据文件格式如下
"""__json__
[
    {
        "messages": [
        {
            "content": "Imagine you are a robot programmed for navigation tasks. You have given a video of historical obervations: <iamge> <image> and current observation: <image>. Your assigned task is <task>. Amalyze this series of images to decide your next move, Which could involve turning left or right by a specific degree, moveing forward a certain distance, or stop if task is completed.",
            "role": "user"
        },
        {
            "content": "action: turn left 90 degrees",
            "role": "assistant"
        },
        {
            "content": "Imagine you are a robot programmed for navigation tasks. You have given a video of historical obervations: <iamge> <image> and current observation: <image>. Your assigned task is <task>. Amalyze this series of images to decide your next move, Which could involve turning left or right by a specific degree, moveing forward a certain distance, or stop if task is completed.",
            "role": "user"
        },
        {
            "content": "action: turn left 90 degrees",
            "role": "assistant"
        }
        ],
        "images": [
        "images/<episode>/episode_0.jpg",
        "images/<episode>/episode_0.jpg",
        ]
    },
    {
        "messages": [
        {
            "content": "Imagine you are a robot programmed for navigation tasks. You have given a video of historical obervations: <iamge> <image> and current observation: <image>. Your assigned task is <task>. Amalyze this series of images to decide your next move, Which could involve turning left or right by a specific degree, moveing forward a certain distance, or stop if task is completed.",
            "role": "user"
        },
        {
            "content": "action: turn left 90 degrees",
            "role": "assistant"
        },
        {
            "content": "Imagine you are a robot programmed for navigation tasks. You have given a video of historical obervations: <iamge> <image> and current observation: <image>. Your assigned task is <task>. Amalyze this series of images to decide your next move, Which could involve turning left or right by a specific degree, moveing forward a certain distance, or stop if task is completed.",
            "role": "user"
        },
        {
            "content": "action: turn left 90 degrees",
            "role": "assistant"
        }
        ],
        "images": [
        "images/<episode>/episode_0.jpg",
        "images/<episode>/episode_0.jpg",
        ]
    }
]
"""



# import os



import argparse

# import json
import os
import time
from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import orjson 

def list_first_level(path):
    return [str(child.absolute()) for child in Path(path).iterdir()]

# print(list_first_level("data"))

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

class Json_Templater:
    def __init__(self):
        self.config = load_yamls_from_dir("utils")
        self.json_file_save_path = self.config.qwen_r2r_dataset_cfg.json_file_save_path
        self.dataset_file_path = self.config.qwen_r2r_dataset_cfg.dataset_file_path
        
        self.template_header_str = f"Imagine you are a robot programmed for navigation tasks. You have given "
        self.temlpate_history_str = f"a video of historical obervations: <history_image> and "
        self.template_current_str = f"current observation: <image>. "
        self.template_task_str = f"Your assigned task is: \"<task>\". Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if task is completed."
        
        self.max_historical_image_num = self.config.qwen_r2r_dataset_cfg.max_historical_image_num

    def create_json_template(self,type):
        # Create the JSON template
        json_data = None
        if(type == "src"):
            json_data = {
                "messages": [],
                "images": []
            }
        elif(type == "navila"):
            json_data = {
                "messages": [
                    {
                        "content": f"Imagine you are a robot programmed for navigation tasks. You have given a video of historical obervations: <iamge> <image> <iamge> <iamge> and current observation: <image>. Your assigned task is: \"<task>\". Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if task is completed.",
                        "role": "user"
                    },
                    {
                        "content": f"action: {self.action_to_str(self.action_list[0])}",
                        "role": "assistant"
                    }
                ],
                "images": []
            }
        
        return json_data
    
    # def merge_action(self, action_list ,upper_limit_action_number):
    #     mergerd_action_list = []
    #     disapper_action_index = []
        
    #     for i in range(len(action_list)):
    #         mergerd_action_list.append(action_list[i])
    #         if i < len(action_list) - upper_limit_action_number:
    #             for j in range(upper_limit_action_number - 1):
    #                 if action_list[i+j+1] == action_list[i]:
    #                     disapper_action_index.append(i+j+1)
    #                     mergerd_action_list[i]  = f"{mergerd_action_list[i]} {action_list[i+j+1]}"
    #                 else:
    #                     i += j
    #         else:
    #             left_action_number = len(action_list) - i
    #             if left_action_number <= 1:
    #             # current_limit = upper_limit_action_number
                
    #     return mergerd_action_list, disapper_action_index
    
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

    def fill_in_json_template(self):
        folder_list = list_first_level(self.dataset_file_path)
        r2r_data_set_json = []
        folder_count = 0
        start_time = time.time()
        current_time = time.time()
        end_time = time.time()
        for folder in folder_list:
            folder_count += 1
            if(folder_count % 100 == 0):
                end_time = time.time()
                duration = end_time - current_time
                totol_duration = end_time - start_time
                current_time = time.time()
                print(f"{folder_count} / {len(folder_list)} , {duration} , {totol_duration}")
            
            episode_id = folder.split("/")[-1]
            images_save_path_list = list_first_level(folder)
            
            instruction_str = ""
            action_str = ""
            
            instruction_save_path  = os.path.join(folder, f"{episode_id}_instruction.txt")
            with open(instruction_save_path, "r") as f:
                instruction_str = f.read()
            
            action_save_path = os.path.join(folder, f"{episode_id}_actions.txt")
            with open(action_save_path, "r") as f:
                action_str = f.read()
            action_list = action_str.split(", ")
            # print(action_list)
            
            merged_actions, merged_indices = self.merge_actions(action_list, 3)
            # print(merged_actions)
            
            # 单个folder中，单条轨迹数据转换的json缓存在这里
            json_data= self.create_json_template("src")
            
            # 每个动作生成一条数据
            for step in range(len(merged_actions)):
                action_str = self.action_to_str(merged_actions[step])
                # print(action_str)

                json_assistant = {
                    "messages": [{
                        "content": f"{action_str}",
                        "role": "assistant"
                        }
                    ]
                }
                
                header_str = ""
                history_str = ""
                current_str = ""
                task_str = ""
                
                # 不用绝对路径了，改成相对路径，因为docker里面找路径和主机不一样
                image_list = []

                if(step == 0):
                    header_str = self.template_header_str
                    current_str = self.template_current_str
                    task_str = f'{self.template_task_str}'.replace("<task>", instruction_str)
                    image_list.append(f"r2r_envdrop/{episode_id}/{episode_id}_{merged_indices[step][0]}.jpg")
                    # image_list.append(f"{folder}/{episode_id}_{merged_indices[step][0]}.jpg")
                elif(step <= self.max_historical_image_num):
                    # historical_image_str = "<image>"
                    historical_image_str = ""
                    for i in range(step):
                        historical_image_str += f" <image>"
                    header_str = self.template_header_str
                    history_str = self.temlpate_history_str.replace("<history_image>",historical_image_str)
                    current_str = self.template_current_str
                    task_str = self.template_task_str.replace("<task>", instruction_str)
                    for i in range(step):
                        # image_list.append(f"{folder}/{episode_id}_{merged_indices[step-i-1][0]}.jpg")
                        image_list.append(f"r2r_envdrop/{episode_id}/{episode_id}_{merged_indices[step-i-1][0]}.jpg")
                    image_list.reverse()    
                    # image_list.append(f"{folder}/{episode_id}_{merged_indices[step][0]}.jpg")
                    image_list.append(f"r2r_envdrop/{episode_id}/{episode_id}_{merged_indices[step][0]}.jpg")
                else:
                    header_str = self.template_header_str
                    history_str = self.temlpate_history_str.replace("<history_image>","<image> <image> <image> <image>")
                    current_str = self.template_current_str
                    task_str = self.template_task_str.replace("<task>", instruction_str)
                    for i in range(self.max_historical_image_num):
                        # image_list.append(f"{folder}/{episode_id}_{merged_indices[step-i-1][0]}.jpg")
                        image_list.append(f"r2r_envdrop/{episode_id}/{episode_id}_{merged_indices[step-i-1][0]}.jpg")
                    image_list.reverse()
                    # image_list.append(f"{folder}/{episode_id}_{merged_indices[step][0]}.jpg")
                    image_list.append(f"r2r_envdrop/{episode_id}/{episode_id}_{merged_indices[step][0]}.jpg")
                
                json_user = {
                    "messages": [{
                        "content": f"{header_str}{history_str}{current_str}{task_str}",
                        "role": "user"
                        }
                    ],
                    "images": image_list
                }
                

                
                json_data["messages"].extend([json_user["messages"][0], json_assistant["messages"][0]])
                json_data["images"].extend(image_list)
                
            # 将json_data 写入 总的dict
            r2r_data_set_json.append(json_data)
                
        # print(json_data)    
        # save json_data using orjson
        json_bytes = orjson.dumps(r2r_data_set_json, f, option=orjson.OPT_INDENT_2)
        # with open(f"{folder}/{episode_id}.json", "w") as f:
        with open(f"r2r_train.json", "wb") as f:
            f.write(json_bytes)              
        # exit()
    def action_to_str(self, action):
        # Convert action to string
        if action == "0":
            return "stop"
        elif action[0] == "1":
            return f"move forward {0.25*len(action)} meter"
        elif action[0] == "2":
            return f"turn left {15*len(action)} degrees"
        elif action[0] == "3":
            return f"turn right {15*len(action)} degrees"
        else:
            return action
        
    
if __name__ == "__main__":
    json_templater = Json_Templater()
    json_templater.fill_in_json_template()