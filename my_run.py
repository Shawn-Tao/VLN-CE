#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import (
    evaluate_agent,
    nonlearning_inference,
)

from my_scripts.my_agents import qwen_inference, gt_inference_r2r, gt_inference_rxr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference", "qwen", "gt_record_r2r", "gt_record_rxr"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val_seen", "val_unseen", "envdrop"],
        type=str,
        required=True,
        help="data split",
    )
    
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, split:str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    # if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
    if run_type == "qwen":
        # nonlearning_inference(config)
        qwen_inference(config, data_split=split)
        return
    
    if run_type == "gt_record_r2r":
        gt_inference_r2r(config)
        return
    
    if run_type == "gt_record_rxr":
        gt_inference_rxr(config)
        return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    # print(type(trainer_init)) -><class 'type'> content -> DaggerTrainer
    # exit()
    
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    
    trainer = trainer_init(config)
    
    # print(type(config))
    # exit()
    
    with open("run_config.yaml", "w") as f:
        f.write(config.dump())
    
    exit()

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()
    # elif run_type == "make_gt_video":
    #     trainer.make_gt_video()


if __name__ == "__main__":
    main()
