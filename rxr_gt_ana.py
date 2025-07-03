#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions

if TYPE_CHECKING:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


import gzip
import json

class GroundTruthFollower:
    r"""Utility class for extracting the action on the shortest path to the
        goal.

    :param sim: HabitatSim instance.
    :param goal_radius: Distance between the agent and the goal for it to be
            considered successful.
    :param return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    :param stop_on_error: Return stop if the follower is unable to determine a
                          suitable action to take next.  If false, will raise
                          a habitat_sim.errors.GreedyFollowerError instead
    """

    def __init__(self, split_str):
        # self.env = env
        # dataset_gt_filename = "data/datasets/vln/mp3d/r2r/v1/{split}/{split}_gt.json.gz".format(split=env.config.dataset.split)
        dataset_gt_filename = "data/datasets/RxR_VLNCE_v0/{split}/{split}_guide_gt.json.gz".format(split=split_str)
        # print("dataset_gt_filename:",dataset_gt_filename)
        f = gzip.open(dataset_gt_filename, "rt")
        self.deserialized = json.loads(f.read())
        f.close()
        self.action_count = 0
        self.current_episode_id = 0
        pass

    def gt_action_parse(self, episode_id):
        self.current_episode_id = episode_id
        self.action_count = 0
        return self.deserialized[str(episode_id)]['actions']
            
    def get_next_action(self):
        next_action = HabitatSimActions.stop
        if(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 0):
            next_action = HabitatSimActions.stop
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 1):
            next_action = HabitatSimActions.move_forward
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 2):
            next_action = HabitatSimActions.turn_left
        elif(self.deserialized[str(self.current_episode_id)]['actions'][self.action_count] == 3):
            next_action = HabitatSimActions.turn_right
        # print("current_action:",next_action)
        self.action_count += 1
        return next_action


if __name__ == "__main__":
    dataset_gt_filename = "data/datasets/RxR_VLNCE_v0/train/train_guide_gt.json.gz"
    
    f = gzip.open(dataset_gt_filename, "rt")
    deserialized = json.loads(f.read())
    f.close()
    
    print(deserialized['1'].keys())
    
    # # 写入 txt 文件
    # with open("rxr_gt_guide.json", "w") as f:
    #     json.dump(deserialized, f, indent=4)
    
    # print(deserialized)