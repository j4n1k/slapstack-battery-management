from os.path import sep, abspath, join

import hydra
from gymnasium.vector import AsyncVectorEnv
from omegaconf import DictConfig, OmegaConf
import os
import time
from typing import Dict

import pandas as pd
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3 import DQN, SAC
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
import wandb
# from wandb.integration.sb3 import WandbCallback
from custom_callbacks import WandbCallback, MaskableEvalCallback

from experiment_commons import (ExperimentLogger, LoopControl,
                                create_output_str, count_charging_stations, delete_prec_dima, gen_charging_stations,
                                gen_charging_stations_left, get_layout_path, delete_partitions_data,
                                get_partitions_path)
from slapstack import SlapEnv
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.output_converters import FeatureConverterCharging
from slapstack_controls.storage_policies import ClosestOpenLocation, ConstantTimeGreedyPolicy, BatchFIFO
from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy)

