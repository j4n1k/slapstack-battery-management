import heapq as h
import random

import numpy as np

from copy import deepcopy
from typing import Tuple, Dict, List, Set, Union
from sortedcontainers import SortedList
from collections import namedtuple, defaultdict

from stable_baselines3 import DQN, SAC

from slapstack.core import State, SlapCore
from slapstack.core_events import Order, Retrieval
from slapstack.core_state_agv_manager import AGV
from slapstack.core_state_location_manager import LocationManager, \
    LocationTrackers
from slapstack.core_state_zone_manager import ZoneManager
from slapstack.helpers import print_3d_np, AccessDirection, ravel, unravel
from slapstack.interface_templates import ChargingStrategy


class ChargingPolicy(ChargingStrategy):
    def __init__(self, name, init=False):
        super().__init__()
        self.name = name

    def get_action(self, state: State, agv_id=None):
        raise NotImplementedError


class FixedChargePolicy(ChargingPolicy):
    def __init__(self, charging_threshold: int):
        super().__init__(name="FixedCharge")
        self.charging_threshold = charging_threshold

    def get_action(self, state: State, agv_id=None):
        agv: AGV = state.agv_manager.agv_index[agv_id]
        charge_to_full = self.charging_threshold - agv.battery
        charging_duration = charge_to_full / state.agv_manager.charging_rate
        return charging_duration


class DQNChargePolicy(ChargingPolicy):
    def __init__(self, model: DQN):
        super().__init__(name="DQN")
        self.model = model

    def get_action(self, state: np.ndarray, agv_id=None):
        action = self.model.predict(state)
        return action

class SACChargePolicy(ChargingPolicy):
    def __init__(self, model: SAC):
        super().__init__(name="SAC")
        self.model = model

    def get_action(self, state: np.ndarray, agv_id=None):
        action = self.model.predict(state)
        return action

class FullChargePolicy(ChargingPolicy):
    def __init__(self):
        super().__init__(name="FullCharge")

    def get_action(self, state: State, agv_id=None):
        agv: AGV = state.agv_manager.agv_index[agv_id]
        charge_to_full = 100 - agv.battery
        charging_duration = charge_to_full / state.agv_manager.charging_rate
        return charging_duration


class RandomChargePolicy(ChargingPolicy):
    def __init__(self, charging_thresholds: list[int], seed: int):
        super().__init__(name="RandomCharge")
        self.charging_thresholds = charging_thresholds
        self.seed = seed

    def get_action(self, state: State, agv_id=None):
        random.seed(self.seed)
        agv: AGV = state.agv_manager.agv_index[agv_id]
        random_th = random.choice(self.charging_thresholds)
        charge_to_full = random_th - agv.battery
        charging_duration = charge_to_full / state.agv_manager.charging_rate

        return charging_duration
