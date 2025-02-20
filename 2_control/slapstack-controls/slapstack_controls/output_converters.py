from collections import deque
from math import sqrt, inf, hypot

import numpy as np

from slapstack.core_events import Delivery, Retrieval
from slapstack.core_state import State
from slapstack.core_state_location_manager import LocationTrackers, LocationManager
from slapstack.helpers import TravelEventKeys
from slapstack.interface_templates import OutputConverter

class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0.0
        self.epsilon = epsilon

    def update(self, reward):
        self.count += 1
        alpha = 1 / self.count
        delta = reward - self.mean
        self.mean += alpha * delta
        self.var = (1 - alpha) * self.var + alpha * delta**2

    def normalize(self, reward):
        return (reward - self.mean) / (self.var**0.5 + self.epsilon)


class SlidingRewardNormalizer:
    def __init__(self, window_size=200000, epsilon=1e-8):
        self.rewards = deque(maxlen=window_size)
        self.epsilon = epsilon

    def update(self, reward):
        self.rewards.append(reward)

    def normalize(self, reward):
        mean = sum(self.rewards) / len(self.rewards)
        var = sum((r - mean) ** 2 for r in self.rewards) / len(self.rewards)
        return (reward - mean) / (var**0.5 + self.epsilon)


class LegacyOutputConverter:
    def __init__(self, reward_type="average_travel_length", state_modifier=None):
        self.previous_average_travel_length = 0
        self.previous_average_service_time = 0
        self.reward_type = reward_type
        self.state_modifier = state_modifier
        self.n_steps = 0
        self.reward_interval = 5

    def modify_state(self, state: State) -> np.ndarray:
        if self.state_modifier == "storage_matrix_only":
            return state.S.flatten()
        if self.state_modifier == "storage_locations_only":
            return state.S[state.S >= 0]
        if self.state_modifier == "lane_free_space":
            return self.calculate_free_spaces_per_lane(state)
        if self.state_modifier == "free_entropy_dominant":
            return self.get_free_space_entropy_and_dominant(state)
        else:
            return state.concatenate().flatten()

    def calculate_free_spaces_per_lane(self, state):
        free_spaces = []
        for lane, lane_info in \
                state.state_cache.lanes.items():
            free_spaces.append(lane_info['n_free_spaces'])
        return np.array(free_spaces)

    def calculate_reward(self, state: State, action: int, legal_actions: list) \
            -> float:
        reward = 0.0
        if self.reward_type == 'average_travel_length':
            reward = self.calculate_average_travel_length_reward(
                action, legal_actions, state)
        elif self.reward_type == 'average_service_time':
            reward = self.calculate_average_service_time_reward(
                action, legal_actions, state)
        elif self.reward_type == 'distance_traveled_shift_penalty':
            reward = self.calculate_distance_traveled(
                action, legal_actions, state)
        reward = reward + 5.1
        return reward

    def calculate_distance_traveled(self, action, legal_actions, state):
        """this reward takes the distance traveled by Travel events and takes a
        percentage of it as a negative reward. It also takes pallet shifts as
        negative reward since they take a while."""
        warehouse_shape = state.S[:, :, 0].shape
        scale = 1/sqrt(warehouse_shape[0]*warehouse_shape[1])
        reward = 0
        if action not in set(legal_actions):
            reward = -500
        else:
            if self.n_steps % self.reward_interval == 0 and self.n_steps:
                reward = -1 * state.trackers.last_travel_distance * scale
                reward -= state.trackers.number_of_pallet_shifts
                if reward < -5:
                    reward = -5
        self.n_steps += 1
        return reward

    def calculate_average_travel_length_reward(self, action, legal_actions,
                                               state):
        average_travel_length = state.trackers.average_travel_length
        difference = self.previous_average_travel_length - average_travel_length
        # if action not in set(legal_actions):
        #     reward = -500
        # else:
        if self.n_steps % self.reward_interval == 0 and self.n_steps:
            # if measurement got worse
            if average_travel_length > self.previous_average_travel_length:
                if difference < -1:
                    reward = -1
                else:
                    reward = difference
            else:  # if measurement got better
                if difference > 1:
                    reward = 1
                else:
                    reward = difference
        else:
            reward = 0.0
        self.n_steps += 1
        self.previous_average_travel_length = average_travel_length
        if reward < -100:
            print()
        return reward

    def calculate_average_service_time_reward(self, action, legal_actions, state):
        average_service_time = state.trackers.average_service_time
        difference = self.previous_average_service_time - average_service_time
        # if action not in set(legal_actions):
        #     reward = -500
        # else:
        if self.n_steps % self.reward_interval == 0 and self.n_steps:
            if average_service_time > self.previous_average_service_time:  # if measurement got worse
                if difference < -1:
                    reward = -1
                else:
                    reward = difference
            else:  # if measurement got better
                if difference > 1:
                    reward = 1
                else:
                    reward = difference
        else:
            reward = 0.0
        self.n_steps += 1
        self.previous_average_service_time = average_service_time
        return reward

    def get_free_space_entropy_and_dominant(self, state):
        entropy = []
        free_spaces = []
        dominant_skus = []
        for lane, lane_info in \
                state.state_cache.lanes.items():
            entropy.append(lane_info['entropy'])
            free_spaces.append(lane_info['n_free_spaces'])
            dominant_skus.append(lane_info['dominant_sku'])
        return np.array(entropy+free_spaces+dominant_skus)


class FeatureConverter(OutputConverter):
    def __init__(self, feature_list):
        self.flattened_entropies = None
        self.fill_level_per_lane = None
        self.fill_level_per_zone = None
        self.sku_counts = None
        self.feature_list = feature_list

    def on_interrupt(self, data):
        pass

    def init_fill_level_per_lane(self, state: State):
        open_locations = np.array(list(state.location_manager.n_open_locations_per_lane.values()))
        total_locations = np.array(list(state.location_manager.n_total_locations_per_lane.values()))
        self.fill_level_per_lane = 1 - open_locations / total_locations

    def f_get_lanewise_entropy_avg(self, state: State):
        if self.flattened_entropies is None:
            self.flattened_entropies = np.array(list(state.location_manager.lane_wise_entropies.values()))
        return np.average(self.flattened_entropies)

    def f_get_lanewise_entropy_std(self, state: State):
        if self.flattened_entropies is None:
            self.flattened_entropies = np.array(list(state.location_manager.lane_wise_entropies.values()))
        return np.std(self.flattened_entropies)

    def f_get_global_entropy(self, state: State):
        sku_counts = np.array(list(state.location_manager.sku_counts.values()))
        p_x = sku_counts[sku_counts != 0] / np.sum(sku_counts)
        return - np.sum(p_x * np.log2(p_x))

    def f_get_lane_fill_level_avg(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.average(self.fill_level_per_lane)

    def f_get_lane_fill_level_std(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.std(self.fill_level_per_lane)

    def f_get_global_fill_level(self, state: State):
        return 1 - state.location_manager.n_open_locations / state.location_manager.n_total_locations

    def f_get_lane_occupancy(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.average(self.fill_level_per_lane < 1)

    def f_get_n_sku_items_avg(self, state: State):
        if self.sku_counts is None:
            if not state.location_manager.sku_counts:
                self.sku_counts = np.array([1])
            else:
                self.sku_counts = np.array(list(state.location_manager.sku_counts.values()))
        # Normalized by max number of occupied locations by any SKU
        return np.average(self.sku_counts) / np.max(self.sku_counts)

    def f_get_n_sku_items_std(self, state: State):
        if self.sku_counts is None:
            if not state.location_manager.sku_counts:
                self.sku_counts = np.array([1])
            else:
                self.sku_counts = np.array(list(state.location_manager.sku_counts.values()))
        # Normalized by max number of occupied locations by any SKU
        return np.std(self.sku_counts) / np.max(self.sku_counts)

    def f_get_n_sku(self, state: State):
        return state.n_skus

    def f_get_total_pallet_shifts(self, state: State):
        # TODO: Normalize by total time steps.
        return state.trackers.number_of_pallet_shifts

    @staticmethod
    def f_get_queue_len_retrieval_orders(state: State):
        # TODO: Normalize by either n_agvs or n_sinks
        return len(state.trackers.queued_retrieval_orders)

    @staticmethod
    def f_get_queue_len_delivery_orders(self, state: State):
        # TODO: Normalize by either n_agvs or n_sources
        return len(state.trackers.queued_delivery_orders)

    def f_get_n_agvs(self, state: State):
        return state.location_manager.n_agvs

    def f_get_free_agv_ratio(self, state: State):
        # TODO: Should we use an if or add a stabilizing term in denominator?
        return len(state.location_manager.free_agv_positions) / (1e-6 + state.location_manager.n_agvs)

    def f_get_legal_actions_avg(self, state: State):
        # Done: Normalize with total number of storage locations.
        return state.location_manager.n_legal_actions_total / (
                (1e-6 + state.location_manager.n_actions_taken) * state.location_manager.n_total_locations
        )

    def init_fill_level_per_zone(self, state: State):
        zm: 'ZoneManager' = state.location_manager.zone_manager
        open_locations = np.array(list(zm.n_open_locations_per_zone.values()))
        total_locations = np.array(list(zm.n_total_locations_per_zone.values()))
        self.fill_level_per_zone = 1 - open_locations / total_locations

    def f_get_zone_fill_level_avg(self, state: State):
        if self.fill_level_per_zone is None:
            self.init_fill_level_per_zone(state)
        return np.average(self.fill_level_per_zone)

    def f_get_zone_fill_level_std(self, state: State):
        if self.fill_level_per_zone is None:
            self.init_fill_level_per_zone(state)
        return np.std(self.fill_level_per_zone)

    def f_get_legal_actions_std(self, state: State):
        e_x2 = state.location_manager.n_legal_actions_squared_total / (1e-6 + state.location_manager.n_actions_taken)
        e_x_2 = state.location_manager.n_legal_actions_total / (1e-6 + state.location_manager.n_actions_taken)
        return (e_x2 - (e_x_2 ** 2)) / state.location_manager.n_total_locations

    def f_get_legal_actions_current(self, state: State):
        return state.location_manager.n_legal_actions_current / (
                (1e-6 + state.location_manager.n_actions_taken) * state.location_manager.n_total_locations
        )

    def modify_state(self, state: State) -> np.ndarray:
        self.__init__(self.feature_list)

        features = []
        for feature_name in self.feature_list:
            features.append(getattr(self, f'f_get_{feature_name}')(state))

        return np.array(features)

        # Average lane entropy
        # Std dev lane entropy
        # Global entropy
        # Avg lane fill level
        # Std dev lane fill level
        # Global fill level
        # Lane occupancy percentage
        # Avg/Std dev number of items for each SKU / total.
        # Number of SKUs.
        # Global queue length of pending retrieval and delivery orders.
        # Number of each type of AGV.
        # AGV occupancy percentage for each type and over all types.
        # Avg/Stddev number of legal actions
        # Current number of legal actions

        # Total and/or rolling window pallet shifts

        # Avg distance traveled (global or per SKU or/and per AGV)
        # Avg service time (global or per SKU or/and per AGV)
        # Std dev distance traveled

        # Avg number of batches per SKU in the warehouse: FIFO inter-batches, BatchFIFO intra-batches
        # Entrance and exit occupancies: Needs to be implemented in simulation
        # Avg/Stddev age for each SKU
        # Avg/Stddev round trip time for each SKU
        # Loaded and unloaded travel distances (global or/and per AGV)
        # Entropy over agreement of strategies on current action. (possibly)
        # Editing distance of lanes as compared to a baseline.
        # Std dev of storage matrix.
        pass

    def calculate_reward(self, state: State, action: int, legal_actions: list) -> float:
        return 0


class FeatureConverterCharging(OutputConverter):
    def __init__(self, feature_list, decision_mode="charging", reward_setting=1):
        self.max_total_queue = 0
        self.util_last_step = 0
        self.decision_mode = decision_mode
        self.rewards = []
        self.feature_list = feature_list
        self.fill_level_per_lane = None
        self.max_queue_len = 0
        self.avg_delivery_buffer_len = 0
        self.max_delivery_buffer_len = 0
        self.avg_retrieval_buffer_len = 0
        self.max_retrieval_buffer_len = 0
        self.t_window_util = 1000
        self.sum_window_util = 0
        self.util = self.sum_window_util / self.t_window_util
        self.n_observations = 0
        self.n_observations_prev = 0
        self.n_orders_last_step = 0
        self.n_delivery_orders_last_step = 0
        self.n_retrieval_orders_last_step = 0
        self.service_time_last_step = 0
        self.queue_last_step = 0
        self.time_last_step = 0
        self.n_stacks = 30
        self.running_avg = 0
        self.feature_stack = deque(maxlen=self.n_stacks)
        self.reward_setting = reward_setting
        self.max_distance_seen = 0
        self.history_window = 1000
        self.service_time_history = []
        self.trend_window = 6000
        self.normalizer = RewardNormalizer()
        self.sliding_normalizer = SlidingRewardNormalizer()

    def init_fill_level_per_lane(self, state: State):
        lt: LocationTrackers = state.location_manager.location_trackers
        open_locations = np.array(list(lt.n_open_locations_per_lane.values()))
        total_locations = np.array(list(lt.n_total_locations_per_lane.values()))
        try:
            self.fill_level_per_lane = 1 - open_locations / total_locations
        except:
            pass

    @staticmethod
    def f_get_is_break(state: 'State'):
        state_time = state.time
        next_event_peek_time = state.next_main_event_time
        time_delta = next_event_peek_time - state_time
        max_duration = state.agv_manager.max_charging_time_frame
        if (time_delta > max_duration) and (state_time != 0):
            return 1
        else:
            return 0

    @staticmethod
    def f_get_avg_battery(state: State):
        return state.agv_manager.get_average_agv_battery() / 100

    @staticmethod
    def f_get_std_battery(state: State):
        return state.agv_manager.get_std_agv_battery()

    @staticmethod
    def f_get_n_charges(state: State):
        at = state.agv_manager.agv_trackers
        return np.average(list(at.charges_per_agv.values()))

    @staticmethod
    def f_get_n_free_agv(state: State):
        return state.agv_manager.n_free_agvs / state.agv_manager.n_agvs

    @staticmethod
    def f_get_n_working_agvs(state: State):
        n_busy = state.agv_manager.get_n_busy_agvs()
        n_free = state.agv_manager.n_free_agvs
        n_charging = state.agv_manager.get_n_depleted_agvs()
        n_agvs = state.agv_manager.n_agvs
        try:

            assert (n_busy + n_free + n_charging == n_agvs)
        except:
            print(f"Missmatch: busy: {n_busy}, free: {n_free}, "
                  f"charging: {n_charging}, sum: {n_busy + n_free + n_charging}")
        return n_busy / n_agvs

    @staticmethod
    def f_get_n_depleted_agvs(state: State):
        agvm = state.agv_manager
        return agvm.get_n_depleted_agvs() / agvm.n_agvs

    @staticmethod
    def f_get_avg_battery_working(state: State):
        return state.agv_manager.get_average_agv_battery_working() / 100

    @staticmethod
    def f_get_avg_battery_charging(state: State):
        return state.agv_manager.get_average_agv_battery_charging() / 100

    @staticmethod
    def f_get_curr_agv_battery(state: State):
        agv_id = state.current_agv
        if agv_id == None:
            return 0
        else:
            agvm = state.agv_manager
            agv = agvm.agv_index[agv_id]
            # assert agv.battery > 0
            # try:
            #     assert agv.battery >= 20
            # except:
            #     print(f"Error in Converter: Battery below Threshold: {agv.battery} ")
            # if agv.battery < 0:
            #     print()
            return agv.battery / 100

    @staticmethod
    def f_get_time_next_event(state: State):
        return state.time_to_next_event

    @staticmethod
    def f_get_utilization(state: State):
        agvm = state.agv_manager
        return agvm.n_busy_agvs / agvm.n_agvs

    @staticmethod
    def f_get_charging_utilization(state: State):
        agvm = state.agv_manager
        return agvm.n_charging_agvs / agvm.n_agvs

    @staticmethod
    def f_get_agv_id(state: State):
        agv_id = state.current_agv
        if not agv_id:
            return 0
        else:
            return state.current_agv / len(state.agv_manager.agv_index)

    def f_get_dist_to_cs(self, state: State):
        # agv_id = state.current_agv
        # if agv_id == None:
        #     return 0
        # agv = state.agv_manager.agv_index[agv_id]
        # agv_position = agv.position
        # min_distance = inf
        # for cs_position in state.agv_manager.charging_stations:
        #     d = state.agv_manager.router.get_distance(
        #         agv_position,
        #         (cs_position[0], cs_position[1])
        #     )
        #     if d < min_distance:
        #         min_distance = d
        # return min_distance / state.agv_manager.router.max_distance
        agv_id = state.current_agv
        if agv_id is None:
            return 0

        agv = state.agv_manager.agv_index[agv_id]
        agv_position = agv.position
        min_distance = inf
        for cs_position in state.agv_manager.charging_stations:
            d = state.agv_manager.router.get_distance(
                agv_position,
                (cs_position[0], cs_position[1])
            )
            if d < min_distance:
                min_distance = d

        # Update max_distance_seen if we see a larger value
        if min_distance > self.max_distance_seen:
            self.max_distance_seen = min_distance

        # Normalize by max distance seen so far
        if self.max_distance_seen > 0:
            return min_distance / self.max_distance_seen
        return 0.0


    @staticmethod
    def f_get_state_time(state: State):
        return state.time

    @staticmethod
    def f_get_service_time(state: State):
        return state.trackers.average_service_time

    def f_get_lane_fill_level_avg(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.average(self.fill_level_per_lane)

    def f_get_lane_fill_level_std(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.std(self.fill_level_per_lane)

    def f_get_avg_entropy(self, state: State):
        lane_entropies = state.location_manager.lane_wise_entropies
        average_entropy = 0
        for lane, entropy in lane_entropies.items():
            average_entropy += entropy
        return average_entropy / len(lane_entropies)

    @staticmethod
    def f_get_global_fill_level(state: State):
        # lt: LocationTrackers = state.location_manager.location_trackers
        # return 1 - lt.n_open_locations / lt.n_total_locations
        return state.trackers.get_fill_level()

    def f_get_free_cs_available(self, state: State):
        agvm = state.agv_manager
        queue = agvm.queued_charging_events
        for cs in agvm.charging_stations:
            if not queue[cs]:
                return 1
            else:
                if len(queue[cs]) <= 1:
                    return 1
        return 0

    # @staticmethod
    # def f_get_avg_battery_cs(state: State):
    #     agvm = state.agv_manager
    #     cs = agvm.charging_stations[0]
    #     queue = agvm.queued_charging_events[cs]
    #     if queue:
    #         charging_event = queue[0]
    #         curr_battery = charging_event.check_battery_charge(state)
    #         return curr_battery / 100
    #     else:
    #         return 0

    @staticmethod
    def f_get_battery_cs(state: State):
        agvm = state.agv_manager
        queue = agvm.queued_charging_events
        battery_levels = []
        for cs in agvm.charging_stations:
            if not queue[cs]:
                battery_levels.append(0)
            else:
                charging_event = queue[cs][0]
                curr_battery = charging_event.check_battery_charge(state)
                battery_levels.append(curr_battery)
        return np.mean(battery_levels) / 100

    @staticmethod
    def f_get_battery_cs1(state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[0]
        queue = agvm.queued_charging_events[cs]
        if queue:
            charging_event = queue[0]
            curr_battery = charging_event.check_battery_charge(state)
            return curr_battery / 100
        else:
            return 0

    @staticmethod
    def f_get_battery_cs2(state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[1]
        queue = agvm.queued_charging_events[cs]
        if queue:
            charging_event = queue[0]
            curr_battery = charging_event.check_battery_charge(state)
            return curr_battery / 100
        else:
            return 0

    @staticmethod
    def f_get_battery_cs3(state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[2]
        queue = agvm.queued_charging_events[cs]
        if queue:
            charging_event = queue[0]
            curr_battery = charging_event.check_battery_charge(state)
            return curr_battery / 100
        else:
            return 0

    @staticmethod
    def f_get_battery_cs4(state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[3]
        queue = agvm.queued_charging_events[cs]
        if queue:
            charging_event = queue[0]
            curr_battery = charging_event.check_battery_charge(state)
            return curr_battery / 100
        else:
            return 0

    def f_get_queue_len_cs1(self, state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[0]
        queued_agv = 0
        if cs in agvm.travel_to_charging_stations.keys():
            travel_to_cs = agvm.travel_to_charging_stations[cs]
            queued_agv += len(travel_to_cs)
        if cs in agvm.booked_charging_stations.keys():
            queue = agvm.booked_charging_stations[cs]
            queued_agv += len(queue)
        if queued_agv > 0:
            return queued_agv / (agvm.n_agvs / 2)
        else:
            return 0

    def f_get_queue_len_cs2(self, state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[1]
        queued_agv = 0
        if cs in agvm.travel_to_charging_stations.keys():
            travel_to_cs = agvm.travel_to_charging_stations[cs]
            queued_agv += len(travel_to_cs)
        if cs in agvm.booked_charging_stations.keys():
            queue = agvm.booked_charging_stations[cs]
            queued_agv += len(queue)
        if queued_agv > 0:
            return queued_agv / (agvm.n_agvs / 2)
        else:
            return 0

    def f_get_queue_len_cs3(self, state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[2]
        queued_agv = 0
        if cs in agvm.travel_to_charging_stations.keys():
            travel_to_cs = agvm.travel_to_charging_stations[cs]
            queued_agv += len(travel_to_cs)
        if cs in agvm.booked_charging_stations.keys():
            queue = agvm.booked_charging_stations[cs]
            queued_agv += len(queue)
        if queued_agv > 0:
            return queued_agv / (agvm.n_agvs / 2)
        else:
            return 0

    def f_get_queue_len_cs4(self, state: State):
        agvm = state.agv_manager
        cs = agvm.charging_stations[3]
        queued_agv = 0
        if cs in agvm.travel_to_charging_stations.keys():
            travel_to_cs = agvm.travel_to_charging_stations[cs]
            queued_agv += len(travel_to_cs)
        if cs in agvm.booked_charging_stations.keys():
            queue = agvm.booked_charging_stations[cs]
            queued_agv += len(queue)
        if queued_agv > 0:
            return queued_agv / (agvm.n_agvs / 2)
        else:
            return 0

    def f_get_queue_len_charging_station(self, state: State):
        agvm = state.agv_manager
        queue_per_station = {cs: 0 for cs in range(agvm.n_charging_stations)}
        for cs in agvm.booked_charging_stations.keys():
            queue_per_station[cs] = len(agvm.booked_charging_stations[cs])
        if np.average(list(queue_per_station.values())) > self.max_queue_len:
            self.max_queue_len = np.average(list(queue_per_station.values()))
        if self.max_queue_len == 0:
            return 0
        else:
            return np.average(list(
                queue_per_station.values())) / self.max_queue_len

    def f_get_queue_len_retrieval_orders(self, state: State):
        n_ret = state.trackers.n_queued_retrieval_orders
        return n_ret / 1500
        # if n_ret > self.max_retrieval_buffer_len:
        #     self.max_retrieval_buffer_len = n_ret
        # if self.max_retrieval_buffer_len == 0:
        #     return 0
        # else:
        #     return (state.trackers.n_queued_retrieval_orders
        #             / self.max_retrieval_buffer_len)

    def f_get_queue_len_delivery_orders(self, state: State):
        n_del = state.trackers.n_queued_delivery_orders
        return n_del / 20
        # if n_del > self.max_delivery_buffer_len:
        #     self.max_delivery_buffer_len = n_del
        # if self.max_delivery_buffer_len == 0:
        #     return 0
        # else:
        #     return (state.trackers.n_queued_delivery_orders
        #             / self.max_delivery_buffer_len)

    def f_get_throughput_delivery_orders(self, state: State):
        delivery_orders_now = 0
        for order in state.trackers.finished_orders:
            if order.type == "delivery":
                delivery_orders_now += 1
        delta_delivery_orders = (delivery_orders_now -
                                 self.n_delivery_orders_last_step)
        time_delta = state.time - self.time_last_step
        self.n_delivery_orders_last_step = delivery_orders_now
        if time_delta > 0:
            throughput_delivery = delta_delivery_orders / time_delta
            return throughput_delivery
        else:
            return 0

    def f_get_throughput_retrieval_orders(self, state: State):
        retrieval_orders_now = 0
        for order in state.trackers.finished_orders:
            if order.type == "retrieval":
                retrieval_orders_now += 1
        delta_retrieval_orders = (retrieval_orders_now -
                                  self.n_retrieval_orders_last_step)
        time_delta = state.time - self.time_last_step
        self.n_retrieval_orders_last_step = retrieval_orders_now
        if time_delta > 0:
            throughput_retrieval = delta_retrieval_orders / time_delta
            return throughput_retrieval
        else:
            return 0

    @staticmethod
    def f_get_t_next_order(state: State):
        if state.next_e:
            return state.next_e.time - state.next_e.time  #/ 3600
        else:
            return 0

    @staticmethod
    def f_get_hour_sin(state: State):
        """Return sine of hour of day for cyclical representation"""
        seconds_in_day = 24 * 3600
        time_of_day = state.time % seconds_in_day
        hour = (time_of_day / 3600)  # 0-24
        return np.sin(2 * np.pi * hour / 24)

    @staticmethod
    def f_get_hour_cos(state: State):
        """Return cosine of hour of day for cyclical representation"""
        seconds_in_day = 24 * 3600
        time_of_day = state.time % seconds_in_day
        hour = (time_of_day / 3600)  # 0-24
        return np.cos(2 * np.pi * hour / 24)

    @staticmethod
    def f_get_day_of_week(state: State):
        """Return normalized day of week (0-1)"""
        seconds_in_day = 24 * 3600
        day = (state.time // seconds_in_day) % 7
        return day / 7  # normalize to 0-1

    @staticmethod
    def f_get_day_sin(state: State):
        """Return sine of day of week for cyclical representation"""
        seconds_in_day = 24 * 3600
        day = (state.time // seconds_in_day) % 7  # 0-6
        return np.sin(2 * np.pi * day / 7)

    @staticmethod
    def f_get_day_cos(state: State):
        """Return cosine of day of week for cyclical representation"""
        seconds_in_day = 24 * 3600
        day = (state.time // seconds_in_day) % 7  # 0-6
        return np.cos(2 * np.pi * day / 7)

    @staticmethod
    def f_get_util_since_last_charge(state: State):
        agv_id = state.current_agv
        if not agv_id:
            return 0
        else:
            agv = state.agv_manager.agv_index[agv_id]
            return agv.util_since_last_charge #/ state.time if state.time > 0 else 0


    @staticmethod
    def f_get_orders_since_last_charge(state: State):
        agv_id = state.current_agv
        if not agv_id:
            return 0
        else:
            agv = state.agv_manager.agv_index[agv_id]
            if len(state.trackers.finished_orders) == 0:
                return 0
            else:
                return agv.orders_since_last_charge # / len(state.trackers.finished_orders)

    @staticmethod
    def f_get_throughput_since_last_charge(state: State):
        agv_id = state.current_agv
        if not agv_id:
            return 0
        else:
            agv = state.agv_manager.agv_index[agv_id]
            if len(state.trackers.finished_orders) == 0:
                return 0
            elif agv.util_since_last_charge == 0:
                return 0
            else:
                return agv.orders_since_last_charge / agv.util_since_last_charge


    @staticmethod
    def f_get_orders_not_served(state: State):
        pass

    @staticmethod
    def f_get_orders_next_hour(state: State):
        return state.orders_next_hour / 3600

    @staticmethod
    def f_get_travel_time_avg(state: State):
        tes = state.trackers.travel_event_statistics
        return tes.average_travel_time() / 100

    @staticmethod
    def f_get_average_distance(state: State):
        tes = state.trackers.travel_event_statistics
        return tes.average_travel_distance() / 100

    @staticmethod
    def f_get_utilization_time(state: State):
        am = state.agv_manager
        return am.get_average_utilization() / state.time if state.time != 0 else 0

    @staticmethod
    def f_get_travel_time_retrieval_avg(state: State):
        tes = state.trackers.travel_event_statistics
        return tes.get_average_travel_time_retrieval() / 100

    @staticmethod
    def f_get_distance_retrieval_avg(state: State):
        tes = state.trackers.travel_event_statistics
        return tes.get_average_travel_distance_retrieval() / 100

    @staticmethod
    def f_get_next_order_possible(state: State):
        next_e = state.next_e

    @staticmethod
    def f_get_n_pallet_shifts(state: State):
        finished_retrieval = 0
        for o in state.trackers.finished_orders:
            if isinstance(o, Retrieval):
                finished_retrieval += 1

        shifts_avg = state.trackers.number_of_pallet_shifts / finished_retrieval
        return shifts_avg


    def f_get_dist_to_order(self, state: State):
        s = state
        agvm = s.agv_manager
        if state.next_e:
            agv_id = state.current_agv
            agv = state.agv_manager.agv_index[agv_id]
            finished_retrieval = 0
            for o in state.trackers.finished_orders:
                if isinstance(o, Retrieval):
                    finished_retrieval += 1

            shifts_avg = state.trackers.number_of_pallet_shifts / finished_retrieval
            est_shift_time = state.params.shift_penalty * shifts_avg
            # dist_to_next_order = hypot(state.I_O_positions[state.next_e.source][0] - agv.position[0],
            #       state.I_O_positions[state.next_e.source][1] - agv.position[1])
            # time_to_next_order = (dist_to_next_order * state.agv_manager.router.unit_distance
            #                       / state.agv_manager.router.speed)
            # depl_next_order = time_to_next_order / state.params.consumption_rate_unloaded
            if state.agv_manager.get_n_free_agvs() == 0 and state.agv_manager.get_n_depleted_agvs():
                for cq_pos in agvm.queued_charging_events.keys():
                    charging_event_list = agvm.queued_charging_events[cq_pos]
                    if charging_event_list:
                        charging_event = charging_event_list[0]
                        target_battery = charging_event.check_battery_charge(s)

                        try:
                            assert target_battery < 100
                        except:
                            print()
                        if target_battery >= 50:
                            pass

            return shifts_avg
            # if depl_next_order < agv.battery:
            #     return 1
            # else:
            #     return 0
            # else:
            #     return 0
            # if dist_to_next_order > self.max_distance_seen:
            #     self.max_distance_seen = dist_to_next_order

            # Normalize by max distance seen so far
            # if self.max_distance_seen > 0:
            #     return dist_to_next_order / self.max_distance_seen
            # agv_position = agv.position
            # min_distance = inf
            # for cs_position in state.agv_manager.charging_stations:
            #     d = state.agv_manager.router.get_distance(
            #         agv_position,
            #         (cs_position[0], cs_position[1])
            #     )
            #     if d < min_distance:
            #         min_distance = d
            # return min_distance / dist_to_next_order
            # return depl_next_order / agv.battery
        else:
            return 0




    def modify_state(self, state: State) -> np.ndarray:
        if self.decision_mode == "charging_check":
            pass
        if state.agv_manager.agv_trackers.n_charges == 0:
            self.time_last_step = state.time
        features = []
        for feature_name in self.feature_list:
            features.append(getattr(self, f'f_get_{feature_name}')(state))
        # if not np.any(self.feature_stack):
        #     self.feature_stack = np.zeros((self.n_stacks,
        #                                   len(self.feature_list)))
        self.feature_stack.append(features)
        stacked_features = np.array(self.feature_stack)
        if len(self.feature_stack) < self.n_stacks:
            padding_needed = self.n_stacks - len(self.feature_stack)
            padding = np.zeros((padding_needed, len(self.feature_list)))
            stacked_features = np.vstack((padding, stacked_features))
        return np.array(features)
        # return stacked_features

    def reset(self):
        self.n_observations = 0
        self.n_observations_prev = 0
        self.n_orders_last_step = 0
        self.n_delivery_orders_last_step = 0
        self.n_retrieval_orders_last_step = 0
        self.service_time_last_step = 0
        self.queue_last_step = 0
        self.time_last_step = 0
        self.running_avg = 0
        self.rewards = []
        self.util_last_step = 0
        self.feature_stack = deque(maxlen=self.n_stacks)
        # self.service_time_history = []
        # self.normalizer = RewardNormalizer()

    def calculate_reward(self,
                         state: State,
                         action: int,
                         legal_actions: list,
                         decision_mode: str,
                         agv_id=None) -> float:

        if decision_mode == self.decision_mode and not isinstance(action, tuple):
            if self.reward_setting == 1:
                return - state.trackers.average_service_time / 3600

            elif self.reward_setting == 2:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                return - total_queue / 350

            elif self.reward_setting == 3:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                clipped_st = min(3600, state.trackers.average_service_time)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                queue_greater_zero = 1 if total_queue > 0 else 0
                return (- 1 * (clipped_st / 3600)
                        - 1 * (clipped_queue / 350)
                        + (n_free_amr / state.agv_manager.n_agvs) * queue_greater_zero
                        )

            elif self.reward_setting == 4:
                reward = 0
                free_cs = self.f_get_free_cs_available(state)
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                state.last_queue_zero_reward = 0
                state.last_free_cs_reward = 0
                state.last_free_cs_penalty = 0
                state.last_no_amr_penalty = 0
                if total_queue == 0 and action == 1:
                    state.last_queue_zero_reward = 1
                    reward += 1
                if free_cs and action == 1:
                    state.last_free_cs_reward = 1
                    reward += 1
                elif not free_cs and action == 1:
                    state.last_free_cs_penalty = -1
                    reward -= 1
                if state.agv_manager.n_free_agvs == 0 and action == 1:
                    state.last_no_amr_penalty = -1
                    reward -= 1
                return - (total_queue/ 350) + reward

        else:
            return 0

    @staticmethod
    def valid_action_mask(self, state: State):
        agv_id = state.current_agv
        agv = state.agv_manager.agv_index[agv_id]
        battery_level = agv.battery
        charging_thresholds = np.array(state.params.charging_thresholds)
        mask = (charging_thresholds == 0) | (charging_thresholds > battery_level)
        return mask
