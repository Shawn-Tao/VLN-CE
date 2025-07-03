import json
from collections import defaultdict

import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange
# from PIL import Image

import cv2
import imageio
import os
import time
import gzip
import shutil

# from vlnce_baselines.common.environments import VLNCEInferenceEnv
# from vlnce_baselines.common.environments import VLNCEWaypointEnv
from vlnce_baselines.common.environments import VLNCEDaggerEnv

from habitat_extensions import maps
from habitat.utils.visualizations import maps as habitat_maps

import sys
sys.path.append("/home/sy1106/HDD/TZL/VLN/VLN-CE/my_scripts")  # 添加自定义路径
print(sys.path)

from utils.tcp_sender import CommandImageSender

import re


def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    env = Env(config=config.TASK_CONFIG)

    assert config.EVAL.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "EVAL.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.EVAL.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = HandcraftedAgent()

    stats = defaultdict(float)
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    for _ in trange(num_episodes):
        obs = env.reset()
        agent.reset()

        while not env.episode_over:
            action = agent.act(obs)
            obs = env.step(action)

        for m, v in env.get_metrics().items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)


def nonlearning_inference(config: Config) -> None:
    split = config.INFERENCE.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.freeze()

    env = VLNCEInferenceEnv(config=config)

    assert config.INFERENCE.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "INFERENCE.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.INFERENCE.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = HandcraftedAgent()

    episode_predictions = defaultdict(list)
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        env.reset()
        obs = agent.reset()

        episode_id = env.current_episode.episode_id
        episode_predictions[episode_id].append(env.get_info(obs))

        while not env.get_done(obs):
            obs = env.step(agent.act(obs))
            episode_predictions[episode_id].append(env.get_info(obs))

    with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
        json.dump(episode_predictions, f, indent=2)

    logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")


class RandomAgent(Agent):
    """Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}


class HandcraftedAgent(Agent):
    """Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}

class gt_flower_r2r():
    def __init__(self,split="train"):
        # self.env = env
        # self.reset()
        dataset_gt_filename = "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz".format(split=split)
        # print("dataset_gt_filename:",dataset_gt_filename)
        f = gzip.open(dataset_gt_filename, "rt")
        self.deserialized = json.loads(f.read())
        f.close()
        self.action_count = 0
        self.current_episode_id = 0
        
    def gt_action_parse(self, episode_id):
        self.current_episode_id = episode_id
        self.action_count = 0
        return self.deserialized[str(episode_id)]['actions']
    
    def get_next_action(self):
        next_action = HabitatSimActions.STOP
        if(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 0):
            next_action = HabitatSimActions.STOP
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 1):
            next_action = HabitatSimActions.MOVE_FORWARD
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 2):
            next_action = HabitatSimActions.TURN_LEFT
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 3):
            next_action = HabitatSimActions.TURN_RIGHT
        # print("current_action:",next_action)
        self.action_count += 1
        return next_action

def gt_inference_r2r(config: Config) -> None:
    
    IMAGE_DIR = os.path.join("output", "images")
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    
    # split = config.INFERENCE.SPLIT
    split = "train"
    # split = "envdrop"
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    # config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    # config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.PANO_ANGLE_FEATURE_SENSOR.CAMERA_NUM = 1
    config.TASK_CONFIG.TASK.PANO_ROTATIONS = 1
    # config.TASK_CONFIG.TASK.SENSORS = ['INSTRUCTION_SENSOR', 'SHORTEST_PATH_SENSOR', 'VLN_ORACLE_PROGRESS_SENSOR']
    config.TASK_CONFIG.TASK.SENSORS = ['INSTRUCTION_SENSOR']
    # print(config.TASK_CONFIG.TASK.SENSORS)
    # exit()
    config.freeze()
    
    with open("inference_config.yaml", "w") as f:
        f.write(config.dump())  # 默认格式
        
    env = VLNCEDaggerEnv(config=config)
    
    agent = gt_flower_r2r(config.TASK_CONFIG.DATASET.SPLIT)

    episode_predictions = defaultdict(list)
    # stats = defaultdict(float)
    
    start_time = time.time()
    current_start_time = time.time()
    
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        obs = env.reset()
        episode_id = env.current_episode.episode_id
        
        dirname = os.path.join(IMAGE_DIR, f"r2r_{split}", str(episode_id))
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        
        # ! record the ground truth action
        gt_actions = agent.gt_action_parse(episode_id)
        gt_actions_save_path = os.path.join(dirname, f"{episode_id}_actions.txt")
        with open(gt_actions_save_path, "w") as f:
            f.write(', '.join(map(str, gt_actions)))
            
        # ! recort instruction text 
        instruction_str = obs["instruction"]["text"]
        instruction_save_path  = os.path.join(dirname, f"{episode_id}_instruction.txt")
        with open(instruction_save_path, "w") as f:
            f.write(instruction_str)
        
        # ! record first image
        image_count = 0
        im = obs["rgb"]
        image_path = os.path.join(dirname, f"{episode_id}_{image_count}.jpg")
        imageio.imwrite(image_path, im)
        
        while not env.get_done(obs):
            best_action = agent.get_next_action()
            obs = env.step(best_action)
            
            image_count += 1
            im = obs[0]["rgb"]
            image_path = os.path.join(dirname, f"{episode_id}_{image_count}.jpg")
            imageio.imwrite(image_path, im)
            
        
        # for m, v in env.get_info(obs).items():
        #     stats[m] += v
        #     print(m, v)
            
        end_time = time.time()
        elapsed_time = end_time - current_start_time
        totol_time = end_time - start_time
        current_start_time = time.time()
        print(f"current step: {iter}, totol time: {totol_time:.2f} second, time elapsed: {elapsed_time:.2f} seconds")

class gt_flower_rxr():
    def __init__(self,split_str="train"):
        dataset_gt_filename = "data/datasets/RxR_VLNCE_v0/{split}/{split}_guide_gt.json.gz".format(split=split_str)
        # print("dataset_gt_filename:",dataset_gt_filename)
        f = gzip.open(dataset_gt_filename, "rt")
        self.deserialized = json.loads(f.read())
        f.close()
        self.action_count = 0
        self.current_episode_id = 0
        
    def gt_action_parse(self, episode_id):
        self.current_episode_id = episode_id
        self.action_count = 0
        return self.deserialized[str(episode_id)]['actions']
    
    def get_next_action(self):
        next_action = HabitatSimActions.STOP
        if(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 0):
            next_action = HabitatSimActions.STOP
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 1):
            next_action = HabitatSimActions.MOVE_FORWARD
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 2):
            next_action = HabitatSimActions.TURN_LEFT
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 3):
            next_action = HabitatSimActions.TURN_RIGHT
        # print("current_action:",next_action)
        self.action_count += 1
        return next_action
    

def gt_inference_rxr(config: Config) -> None:
    
    IMAGE_DIR = os.path.join("output", "images")
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    
    # split = config.INFERENCE.SPLIT
    split = "train"
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    # config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    # config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.PANO_ANGLE_FEATURE_SENSOR.CAMERA_NUM = 1
    config.TASK_CONFIG.TASK.PANO_ROTATIONS = 1
    # config.TASK_CONFIG.TASK.SENSORS = ['INSTRUCTION_SENSOR', 'SHORTEST_PATH_SENSOR', 'VLN_ORACLE_PROGRESS_SENSOR']
    # print(config.TASK_CONFIG.TASK.SENSORS)
    # exit()
    
    # 默认的这个RXR_INSTRUCTION_SENSOR 带着 text_feature ，对我们来说没啥用
    # config.TASK_CONFIG.TASK.SENSORS =["RXR_INSTRUCTION_SENSOR"]
    config.TASK_CONFIG.TASK.SENSORS =["INSTRUCTION_SENSOR"]
    
    config.freeze()
    
    with open("inference_rxr_config.yaml", "w") as f:
        f.write(config.dump())  # 默认格式
        
    env = VLNCEDaggerEnv(config=config)
    
    agent = gt_flower_rxr(config.TASK_CONFIG.DATASET.SPLIT)

    episode_predictions = defaultdict(list)
    # stats = defaultdict(float)
    
    start_time = time.time()
    current_start_time = time.time()
    
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        obs = env.reset()
        episode_id = env.current_episode.episode_id
        
        dirname = os.path.join(IMAGE_DIR, f"rxr_{split}", str(episode_id))
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        
        # ! record the ground truth action
        gt_actions = agent.gt_action_parse(episode_id)
        gt_actions_save_path = os.path.join(dirname, f"{episode_id}_actions.txt")
        with open(gt_actions_save_path, "w") as f:
            f.write(', '.join(map(str, gt_actions)))
            
        # ! recort instruction text 
        instruction_str = obs["instruction"]["text"]
        instruction_save_path  = os.path.join(dirname, f"{episode_id}_instruction.txt")
        with open(instruction_save_path, "w") as f:
            f.write(instruction_str)
        
        # ! record first image
        image_count = 0
        im = obs["rgb"]
        image_path = os.path.join(dirname, f"{episode_id}_{image_count}.jpg")
        imageio.imwrite(image_path, im)
        
        while not env.get_done(obs):
            best_action = agent.get_next_action()
            obs = env.step(best_action)
            
            image_count += 1
            im = obs[0]["rgb"]
            image_path = os.path.join(dirname, f"{episode_id}_{image_count}.jpg")
            imageio.imwrite(image_path, im)
            
        
        # for m, v in env.get_info(obs).items():
        #     stats[m] += v
        #     print(m, v)
            
        end_time = time.time()
        elapsed_time = end_time - current_start_time
        totol_time = end_time - start_time
        current_start_time = time.time()
        print(f"current step: {iter}, totol time: {totol_time:.2f} second, time elapsed: {elapsed_time:.2f} seconds")
    
    


class QwenVLMAgent:
    def __init__(self):
        self.current_step = 0
        self.config_image_list_upper_limit = 9
        self.image_cycle_list = []
        self.episode_id = 1
        
        self.sender = CommandImageSender("127.0.0.1", 8888)
        if not self.sender.connect():
            exit()
        
    def reset(self, current_episode_id):
        self.current_step = 0
        self.image_cycle_list = []
        self.episode_id = current_episode_id
        
    def parse_obs(self, observatioons):
        # return 
        pass
    def act(self, observations):
        
        # inst,images = self.parse_obs(observations)
        if(self.current_step == 0):
            # send instruction and images
            instruction = observations["instruction"]["text"]
            im = observations["rgb"]
            self.image_cycle_list = [im]
            
            image_path = os.path.join("/home/sy1106/HDD/TZL/VLN/VLN-CE/test_image_path", f"{self.current_step}_{0}.jpg")
            imageio.imwrite(image_path, self.image_cycle_list[0])
            
            self.sender.set_command(instruction)
            self.sender.send_images(self.image_cycle_list, include_command=True)
        else:
            im = observations["rgb"]
            self.image_cycle_list.append(im)
            if len(self.image_cycle_list) > self.config_image_list_upper_limit + 1:
                self.image_cycle_list.pop(0)
                
            for i in range(len(self.image_cycle_list)):
                image_path = os.path.join("/home/sy1106/HDD/TZL/VLN/VLN-CE/test_image_path", f"{self.current_step}_{i}.jpg")
                imageio.imwrite(image_path, self.image_cycle_list[i])
            
            # sender.set_command(instruction)
            self.sender.send_images(self.image_cycle_list, include_command=False)
            
        action_str = self.sender.get_aciton_str().get("message")
        
        if self.validate_command(action_str):
            self.current_step += 1
            action,step = self.parse_action_str(action_str)
            return action, step
        else:
            print(f"Invalid command: {action_str}")
            return None, 0
        
        
    
    def validate_command(self, command):
        pattern = r'^(?:move forward (?:\.\d+|\d+(?:\.\d*)?) meters|turn left \d+ degrees|turn right \d+ degrees|stop)$'
        return bool(re.match(pattern, command))
    
    # 需要修改，增加对数字的正则表达
    def parse_action_str(self, action_str:str):
        
        move_match = re.search(r'move forward', action_str)
        if(move_match != None):
            meter_match = re.search(r'meter', action_str)
            distance_str = action_str[move_match.end()+1:meter_match.start()-1]
            distance = float(distance_str)
            # print("move: ", distance)
            num_step = distance/0.25
            return HabitatSimActions.MOVE_FORWARD, num_step
        else:
            turn_left_match = re.search(r'turn left', action_str)
            turn_right_match = re.search(r'turn right', action_str)
            if(turn_left_match != None):
                degree_match = re.search(r'degrees', action_str)
                degree_str = action_str[turn_left_match.end()+1:degree_match.start()-1]
                degree = float(degree_str)
                # print("turn: ", degree)
                num_step = degree/15
                return HabitatSimActions.TURN_LEFT, num_step
            elif(turn_right_match != None):
                degree_match = re.search(r'degrees', action_str)
                degree_str = action_str[turn_right_match.end()+1:degree_match.start()-1]
                degree = float(degree_str)
                # print("turn: ", degree)
                num_step = degree/15
                return HabitatSimActions.TURN_RIGHT, num_step
            else:
                stop_match = re.search(r'stop', action_str)
                # print("stop")
                return HabitatSimActions.STOP, 1


def qwen_inference(config: Config, data_split:str) -> None:    
    # split = config.INFERENCE.SPLIT
    # split = "val_seen"
    # split = "train"
    split = data_split
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    # config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    # config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    # config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.PANO_ANGLE_FEATURE_SENSOR.CAMERA_NUM = 1
    config.TASK_CONFIG.TASK.PANO_ROTATIONS = 1
    # config.TASK_CONFIG.TASK.SENSORS = ['INSTRUCTION_SENSOR', 'SHORTEST_PATH_SENSOR', 'VLN_ORACLE_PROGRESS_SENSOR']
    # print(config.TASK_CONFIG.TASK.SENSORS)
    # exit()
    # TOP_DOWN_MAP_VLNCE
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

    if "RXR_INSTRUCTION_SENSOR" in config.TASK_CONFIG.TASK.SENSORS:
        # 去掉 RXR_INSTRUCTION_SENSOR
        config.TASK_CONFIG.TASK.SENSORS.remove("RXR_INSTRUCTION_SENSOR")
        # 添加 INSTRUCTION_SENSOR
        config.TASK_CONFIG.TASK.SENSORS.append("INSTRUCTION_SENSOR")
        
    if "INSTRUCTION_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
        # 添加 INSTRUCTION_SENSOR
        config.TASK_CONFIG.TASK.SENSORS.append("INSTRUCTION_SENSOR")
    
    config.freeze()
    
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    # 创建基础保存目录
    output_dir = os.path.join("output", f"inference_{time_stamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # with open("inference_config.yaml", "w") as f:
    #     f.write(config.dump())  # 默认格式
        
    env = VLNCEDaggerEnv(config=config)
    
    agent = QwenVLMAgent()
    
    
    # print("start valid*******************:")
    
    # command = "turn up 45 degrees"
    # print(agent.validate_command(command))
    
    # exit()

    episode_predictions = defaultdict(list)
    stats = defaultdict(float)
    
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        obs = env.reset()
        episode_id = env.current_episode.episode_id
        
        # image_path = os.path.join(dirname, f"{episode_id}_{image_count}.jpg")
        # image_path = "output.jpg"
        # imageio.imwrite(image_path, obs['rgb'])
        # print(obs["instruction"]["text"])
        # exit()
        
        agent.reset(episode_id)

        # episode_predictions[episode_id].append(env.get_info(obs))

        # 包含正则表达式，判断返回指令是否符合要求
        action,step = agent.act(obs)

        while(action is None):
            print("last action is none, try inference again")
            action,step = agent.act(obs)
        
        # episode_predictions[episode_id].append(env.get_info(obs))
        # episode_predictions[episode_id].append(env.get_info(obs))
        # print(episode_predictions[episode_id])
        
        while not env.get_done(obs):
            for i in range(int(step)-1):
                obs = env.step(action)
            obs = env.step(action)
            action,step = agent.act(obs[0])

            while(action is None):
                print("last action is none, try inference again")
                action,step = agent.act(obs[0])
            
            # episode_predictions[episode_id].append(env.get_info(obs))
            
            # print(episode_predictions[episode_id])
            if(agent.current_step > 100 and not env.get_done(obs)):
                print("Reach max step limit, break")
                obs = env.step(HabitatSimActions.STOP)
                # break
            
            # # 到达目标，强制停止 
            # if(env.get_info(obs).items()["distance_to_goal"] < 3.0 and not env.get_done(obs)):
            #     print("Reach goal, break")
            #     obs = env.step(HabitatSimActions.STOP)
        
        # for m, v in env.get_metrics().items():
        for m, v in env.get_info(obs).items():
            if(m != "top_down_map_vlnce"):
                stats[m] += v
                print(m, v)
            else:
                # print("top_down_map_vlnce:", v)
                top_down_map = v['map']
                
                top_down_map = maps.colorize_topdown_map(
                    top_down_map, v["fog_of_war_mask"]
                )
                top_down_map = habitat_maps.draw_agent(
                    image=top_down_map,
                    agent_center_coord=v["agent_map_coord"],
                    agent_rotation=v["agent_angle"],
                    agent_radius_px=min(top_down_map.shape[0:2]) // 24,
                )
                
                # print("top_down_map shape:", top_down_map.shape)
                # 将 uint8 类型的 top_down_map 进行处理
                
                imageio.imwrite(f"{output_dir}/top_down_map_vlnce_{episode_id}.jpg", top_down_map)
            
    # with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
    #     json.dump(episode_predictions, f, indent=2)

    # logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")
    
    stats = {k: v / len(env.episodes) for k, v in stats.items()}
    
    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    