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
# from slapstack.core_state_location_manager import LocationManager, \
#     LocationTrackers
from slapstack.core_state_zone_manager import ZoneManager
from slapstack.helpers import print_3d_np, AccessDirection, ravel, unravel
from slapstack.interface_templates import ChargingStrategy


class ChargingPolicy(ChargingStrategy):
    def __init__(self, name, init=False):
        super().__init__()
        self.name = name
        self.type = "charging"

    def get_action(self, state: State, agv_id=None):
        raise NotImplementedError


class LowTHChargePolicy(ChargingStrategy):
    def __init__(self, lower_threshold: float):
        super().__init__()
        self.lower_threshold = lower_threshold
        self.name = "lowth"
        self.type = "go_charging"

    def get_action(self, state: 'State', agv_id: int) -> int:
        # go_charging = state.agv_manager.charge_needed(False, agv_id)
        agv = state.agv_manager.agv_index[agv_id]
        go_charging = agv.battery <= 20
        if go_charging:
            return 1
        else:
            return 0


class DummyChargePolicy(ChargingStrategy):
    def __init__(self,name="opportunity"):
        super().__init__()
        self.name = "dummy+"
        self.type = "go_charging"
        self.charging_threshold = self.name

    def get_action(self, state: 'State', agv_id: int) -> int:
        return 0

class OpportunityPlusChargePolicy(ChargingStrategy):
    def __init__(self,name="opportunity"):
        super().__init__()
        self.name = "opportunity+"
        self.type = "go_charging"
        self.charging_threshold = self.name
    @staticmethod
    def is_break(state: 'State'):
        state_time = state.time
        next_event_peek_time = state.next_main_event_time
        time_delta = next_event_peek_time - state_time
        if time_delta > 0:
            print(time_delta)
        max_duration = state.agv_manager.max_charging_time_frame
        if (time_delta > max_duration) and (state_time != 0):
            # if state.agv_manager.charge_in_break_started == False:
            #     state.agv_manager.full_time_delta = time_delta
            #     state.agv_manager.time_of_next_main_event = next_event_peek_time
            #     state.agv_manager.charge_in_break_started = True
            return True
        else:
            return False

    def get_action(self, state: 'State', agv_id: int) -> int:
        agvm = state.agv_manager
        agv = agvm.agv_index[agv_id]
        queue = agvm.queued_charging_events
        if agv.battery >= 100:
            return 0
        if agv.battery <= 20:
            return 1
        if (state.trackers.n_queued_retrieval_orders == 0 and
                state.trackers.n_queued_delivery_orders == 0):
            # for cs in agvm.booked_charging_stations:
            #     if not agvm.booked_charging_stations[cs]:
            #         return 1
            #     else:
            #         if len(agvm.booked_charging_stations[cs]) <= 1:
            #             return 1
            for cs in agvm.charging_stations:
                if not queue[cs]:
                    return 1
                else:
                    if len(queue[cs]) <= 1:
                        return 1
        return 0



class OpportunityChargePolicy(ChargingStrategy):
    def __init__(self, name="opportunity"):
        super().__init__()
        self.name = name
        self.type = "go_charging"
        self.charging_threshold = self.name

    @staticmethod
    def is_break(state: 'State'):
        state_time = state.time
        next_event_peek_time = state.next_main_event_time
        time_delta = next_event_peek_time - state_time
        if time_delta > 0:
            print(time_delta)
        max_duration = state.agv_manager.max_charging_time_frame
        if (time_delta > max_duration) and (state_time != 0):
            # if state.agv_manager.charge_in_break_started == False:
            #     state.agv_manager.full_time_delta = time_delta
            #     state.agv_manager.time_of_next_main_event = next_event_peek_time
            #     state.agv_manager.charge_in_break_started = True
            return True
        else:
            return False

    def get_action(self, state: 'State', agv_id: int) -> int:
        agvm = state.agv_manager
        agv = agvm.agv_index[agv_id]
        queue = agvm.queued_charging_events
        if agv.battery >= 100:
            return 0
        if agv.battery <= 20:
            return 1
        if (state.trackers.n_queued_retrieval_orders == 0 and
                state.trackers.n_queued_delivery_orders == 0):
            # for cs in agvm.booked_charging_stations:
            #     if not agvm.booked_charging_stations[cs]:
            #         return 1
            #     else:
            #         if len(agvm.booked_charging_stations[cs]) <= 1:
            #             return 1
            for cs in agvm.charging_stations:
                if not queue[cs] and state.agv_manager.n_free_agvs - 1 > 0:
                    return 1
                # else:
                #     if len(queue[cs]) <= 1:
                #         return 1
            # return 1
        # else:
        return 0
        # return 0


class CombinedChargingPolicy(ChargingStrategy):
    def __init__(self, lower_threshold: float, upper_threshold: float, name="FixedCharge"):
        super().__init__()
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.name = name
        self.type = "go_charging"

    @staticmethod
    def _queue_len(state: 'State'):
        return (state.trackers.n_queued_retrieval_orders +
                state.trackers.n_queued_delivery_orders)

    def opportunistic(self, state: 'State', agv_id):
        agvm = state.agv_manager
        agv = agvm.agv_index[agv_id]
        queue = agvm.queued_charging_events
        if agv.battery >= self.upper_threshold:
            return 0
        if agv.battery <= self.lower_threshold:
            print(agv.battery)
            return 1
        if self._queue_len(state) == 0:
            for cs in agvm.charging_stations:
                if not queue[cs] and state.agv_manager.n_free_agvs - 1 > 0:
                    return 1
        return 0

    def fixed(self, agv: AGV):
        if agv.battery <= self.lower_threshold:
            print(agv.battery)
            return True
        else:
            return False

    def get_action(self, state: 'State', agv_id: int) -> int:
        go_charging = False
        agv: AGV = state.agv_manager.agv_index[agv_id]
        if self.name == "FixedCharge" or self.name == "Fixed" or self.name == "HighLow":
            go_charging = self.fixed(agv)
        elif self.name == "Opportunistic":
            go_charging = self.opportunistic(state, agv_id)
        else:
            print("go charging strategy not specified")
        if go_charging:
            if self.name == "HighLow":
                if self._queue_len(state) > 0:
                    charge_to_full = self.upper_threshold - agv.battery
                else:
                    charge_to_full = 100 - agv.battery
            else:
                charge_to_full = self.upper_threshold - agv.battery
            charging_duration = charge_to_full / state.agv_manager.charging_rate
            return charging_duration
        else:
            return 0


class FixedChargePolicy(ChargingPolicy):
    def __init__(self, charging_threshold: int):
        super().__init__(name="FixedCharge")
        self.charging_threshold = charging_threshold
        self.upper_threshold = charging_threshold

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
        # random.seed(self.seed)
        agv: AGV = state.agv_manager.agv_index[agv_id]
        random_th = random.choice(self.charging_thresholds)
        charge_to_full = random_th - agv.battery
        charging_duration = charge_to_full / state.agv_manager.charging_rate

        return charging_duration
