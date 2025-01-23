from collections import deque
from math import sqrt, inf, hypot

import numpy as np

from slapstack.core_events import Delivery, Retrieval
from slapstack.core_state import State
from slapstack.core_state_location_manager import LocationTrackers, LocationManager
from slapstack.helpers import TravelEventKeys
from slapstack.interface_templates import OutputConverter


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

    def calculate_reward(self,
                         state: State,
                         action: int,
                         legal_actions: list,
                         decision_mode: str,
                         agv_id=None) -> float:
        # if state.agv_manager.agv_trackers.n_charges == 0:
        #     self.n_observations = 0
        #     self.n_observations_prev = 0
        #     self.n_orders_last_step = 0
        #     self.n_delivery_orders_last_step = 0
        #     self.n_retrieval_orders_last_step = 0
        #     self.service_time_last_step = 0
        #     self.time_last_step = 0
        #     self.n_stacks = 3
        #     self.feature_stack = np.zeros((0, 0))

        if decision_mode == self.decision_mode and not isinstance(action, tuple):
            if isinstance(action, list):
                action, interrupt_action = action
                interrupt_action -= 1
            else:
                interrupt_action = -1
            # Reward 0
            # return - state.trackers.average_service_time

            #Reward 1
            # if state.agv_manager.agv_trackers.n_charges == 0:
            #     self.n_observations = 0
            #     self.n_observations_prev = 0
            #     self.n_orders_last_step = 0
            #     self.n_delivery_orders_last_step = 0
            #     self.n_retrieval_orders_last_step = 0
            #     self.service_time_last_step = 0
            #     self.time_last_step = 0
            #     self.n_stacks = 3
            #     self.feature_stack = np.zeros((0, 0))
            if self.reward_setting == 0:
                # if state.done:
                # clipped_st = max(state.trackers.average_service_time, 3600)
                # clipped_st = min(15000, state.trackers.average_service_time)
                # return - clipped_st
                # total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                # return - total_queue
                current_service_time = state.trackers.average_service_time
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                # self.n_orders_last_step = n_orders_now
                # return delta_service_time * 10
                return delta_service_time

            if self.reward_setting == 1:
                # if state.done:
                # clipped_st = max(state.trackers.average_service_time, 3600)
                clipped_st = min(8000, state.trackers.average_service_time)
                return - state.trackers.average_service_time
                # else:
                #     return 0
                # return 1 - (state.trackers.average_service_time / 600)
                # agv_id = state.current_agv
                # agv = state.agv_manager.agv_index[agv_id]
                # total_queue = state.trackers.n_queued_retrieval_orders + state.trackers.n_queued_delivery_orders
                # if state.done and state.incomplete_orders:
                #     return -200
                # elif state.done and not state.incomplete_orders:
                #     return 200
                # if total_queue > 0 and state.agv_manager.get_n_depleted_agvs() > state.agv_manager.n_agvs / 4:
                #     return -10
                # if action == 1:
                    # if agv.battery >= 80:
                    #     return -10
                    # else:
                    # 1 base subtract queue ratio (charging good at low ratio add free agv ratio (charging good when idle state)
                #     return 1 - (total_queue / 330) + (state.agv_manager.get_n_free_agvs() / state.agv_manager.n_agvs)
                # else:
                    # if agv.battery <= 20:
                    #     return -10
                    # else:
                    # battery level (dont charge with high battery) + working agv ratio (dont charge when busy state)
                    # return (agv.battery / 100) + (state.agv_manager.get_n_busy_agvs() / state.agv_manager.n_agvs)
            # elif self.reward_setting == 2:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     clipped_queue = min(350, total_queue)
            #     return - (clipped_queue / 350) #* 10
            # elif self.reward_setting == 1:
            #     agv_id = state.current_agv
            #     agv = state.agv_manager.agv_index[agv_id]
            #     penalty = 0
            #     # if agv.battery < 20:
            #     #     penalty = -10
            #     self.n_observations += 1
            #     current_service_time = state.trackers.average_service_time
            #     delta_service_time = self.service_time_last_step - current_service_time
            #     self.service_time_last_step = current_service_time
            #     self.running_avg += delta_service_time
            #     state.running_avg = self.running_avg
            #     self.rewards.append(delta_service_time)
            #     state.rewards = self.rewards
            #     # self.n_orders_last_step = n_orders_now
            #     return delta_service_time #+ penalty
            #Reward 2
            elif self.reward_setting == 2:
                min_delta = -10  # Minimum allowable delta (negative change)
                max_delta = 1 # 1
                # if state.agv_manager.agv_trackers.n_charges == 0:
                #     self.service_time_last_step = state.trackers.average_service_time
                #     return 0
                current_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, current_queue)
                current_service_time = state.trackers.average_service_time
                clipped_st = min(3600, current_service_time)
                delta_service_time = self.service_time_last_step - current_service_time
                delta_queue = self.queue_last_step - clipped_queue #/ 350
                self.queue_last_step = clipped_queue
                self.service_time_last_step = current_service_time
                clipped_delta = max(min(delta_service_time * 10, max_delta), min_delta)
                # self.n_orders_last_step = n_orders_now
                # return delta_service_time * 10
                return delta_queue - (clipped_st / 3600) #delta_service_time +  # - (clipped_queue / 350)

            # elif self.reward_setting == 3:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     clipped_queue = min(350, total_queue)
            #     clipped_st = min(3600, state.trackers.average_service_time)
            #     return (- (clipped_st / 3600)
            #             - (clipped_queue / 350)
            #             )
            # elif self.reward_setting == 3:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
            #     queue_greater_zero = 1 if total_queue > 0 else 0
            #     return - (n_depleted_amr / state.agv_manager.n_agvs) * queue_greater_zero

            # elif self. reward_setting == 3:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     n_free_amr = state.agv_manager.get_n_free_agvs()
            #     queue_greater_zero = 1 if total_queue > 0 else 0
            #     return (n_free_amr / state.agv_manager.n_agvs) * queue_greater_zero

            elif self.reward_setting == 3:
                min_delta = -10  # Minimum allowable delta (negative change)
                max_delta = 1
                current_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders

                # if state.agv_manager.agv_trackers.n_charges == 0:
                #     self.queue_last_step = current_queue
                #     return 0

                current_service_time = state.trackers.average_service_time
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                delta_queue = self.queue_last_step - current_queue
                self.queue_last_step = current_queue
                clipped_delta = max(min(delta_queue * 10, max_delta), min_delta)
                # self.n_orders_last_step = n_orders_now
                # # return delta_service_time * 10
                # if state.done:
                #     return - state.trackers.average_service_time
                return delta_queue #+ delta_service_time

            elif self.reward_setting == 4:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                clipped_st = min(3600, state.trackers.average_service_time)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                queue_greater_zero = 1 if total_queue > 0 else 0
                return (- (clipped_st / 3600)
                        - (clipped_queue / 350)
                        # - (n_depleted_amr / state.agv_manager.n_agvs) * queue_greater_zero
                        + (n_free_amr / state.agv_manager.n_agvs) * queue_greater_zero
                        # - (state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs) * queue_greater_zero
                        # + ((state.agv_manager.get_average_agv_battery() / 100) * clipped_queue / 350)
                        )

            elif self.reward_setting == 5:
                agv = state.agv_manager.agv_index[state.previous_agv]
                agv_battery_level = agv.battery
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                n_free_amr = state.agv_manager.get_n_free_agvs()
                queue_greater_zero = 1 if total_queue > 0 else 0
                if action > 0:
                    if total_queue > 0:
                        return -1
                    elif total_queue == 0:
                        return 1
                elif action == 0:
                    if total_queue > 0:
                        return 1
                    elif total_queue == 0:
                        return -1

            elif self.reward_setting == 6:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(total_queue, 350)
                clipped_st = min(3600, state.trackers.average_service_time)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                n_working_amr = state.agv_manager.get_n_busy_agvs()
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                queue_greater_zero = 1 if total_queue > 0 else 0
                queue_zero = 1 if total_queue == 0 else 0
                # return - 2 * (clipped_queue/ 350) - (n_depleted_amr / state.agv_manager.n_agvs) * (clipped_queue / 350) + state.agv_manager.get_average_agv_battery() / 100
                # return (n_free_amr + n_working_amr) * (clipped_queue / 350) + state.agv_manager.get_average_agv_battery() / 100
                return -(clipped_queue / 350) + (n_free_amr + n_working_amr) * (clipped_queue / 350) + n_depleted_amr * queue_zero

            elif self.reward_setting == 7:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                clipped_st = min(3600, state.trackers.average_service_time)
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                return (
                        - (clipped_queue / 350)
                        - (clipped_st / 3600)
                        - (n_depleted_amr / state.agv_manager.n_agvs) * (clipped_queue / 350)
                    # - (clipped_queue / 350)
                        )
            # elif self.reward_setting == 5:
            #     agv_id = state.current_agv
            #     if not agv_id:
            #         return 0
            #     else:
            #         self.n_observations += 1
            #         current_util = state.agv_manager.get_average_utilization() / state.time
            #         delta_util = self.util_last_step - current_util
            #         self.util_last_step = current_util
            #         self.running_avg += delta_util
            #         state.running_avg = self.running_avg
            #         self.rewards.append(delta_util)
            #         state.rewards = self.rewards
            #         # self.n_orders_last_step = n_orders_now
            #         return delta_util
            # elif self.reward_setting == 6:
            #     agv_id = state.current_agv
            #     if not agv_id:
            #         return 0
            #     else:
            #         agv = state.agv_manager.agv_index[agv_id]
            #         return agv.util_since_last_charge / state.time if state.time > 0 else 0

            # elif self.reward_setting == 7:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     n_depleted_ratio = state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs
            #
            #     # Use a fixed reference value instead of dynamic max
            #     queue_capacity = 400  # or whatever makes sense for your system
            #     queue_ratio = total_queue / queue_capacity
            #
            #     # Inverse relationship: reward depleted AGVs more when queue is small
            #     if total_queue == 0:
            #         # Maximum reward for charging during empty queue
            #         if action == 0:
            #             reward = -1
            #     else:
            #         # Decrease reward as queue grows
            #         reward = 1
            #
            #     # Optionally add penalty for high queue + high depletion
            #     if queue_ratio > 0.3 and n_depleted_ratio > 0.3:
            #         reward -= (queue_ratio * n_depleted_ratio)
            #
            #     return reward
            #
            # elif self.reward_setting == 8:
            #     return - state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs
            elif self.reward_setting == 9:
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
                return reward

            # elif self.reward_setting == 9:
            #     current_service_time = state.trackers.average_service_time
            #     delta_service_time = self.service_time_last_step - current_service_time
            #     self.service_time_last_step = current_service_time
            #     self.service_time_history.append(current_service_time)
            #
            #     # Only keep trend_window entries
            #     if len(self.service_time_history) > self.trend_window:
            #         self.service_time_history.pop(0)
            #
            #
            #     if len(self.service_time_history) < self.trend_window:
            #         return delta_service_time
            #     else:
            #         x = np.arange(len(self.service_time_history))
            #         y = np.array(self.service_time_history)
            #         slope, _ = np.polyfit(x, y, 1)
            #
            #         # Normalize by mean and invert (negative slope = improvement = positive reward)
            #         normalized_trend = -slope / np.mean(y)
            #         return delta_service_time - normalized_trend
            #         # return delta_service_time - slope
            elif self.reward_setting == 10:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                clipped_st = min(3600, state.trackers.average_service_time)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                queue_greater_zero = 1 if total_queue > 0 else 0
                return (- (clipped_queue / 350)
                        # - (n_depleted_amr / state.agv_manager.n_agvs) * queue_greater_zero
                        + (n_free_amr / state.agv_manager.n_agvs) * queue_greater_zero
                        )

            elif self.reward_setting == 11:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                clipped_st = min(3600, state.trackers.average_service_time)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                n_busy_amr = state.agv_manager.get_n_busy_agvs()
                queue_greater_zero = 1 if total_queue > 0 else 0
                return (- (clipped_queue / 350)
                        # - (clipped_st / 3600)
                        # - (n_depleted_amr / state.agv_manager.n_agvs) * queue_greater_zero
                        + ((n_free_amr + n_busy_amr)/ state.agv_manager.n_agvs) * queue_greater_zero
                        # + state.agv_manager.get_average_agv_battery() / 100
                        )

            elif self.reward_setting == 12:
                agv = state.agv_manager.agv_index[state.previous_agv]
                agv_battery_level = agv.battery
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                amr_service = 1 if clipped_queue > 0 and action == 0 else 0
                depletion_penalty = -1 if agv_battery_level <= 20 else 0
                return - clipped_queue / 350 + amr_service - depletion_penalty

            elif self.reward_setting == 13:
                agv = state.agv_manager.agv_index[state.previous_agv]
                agv_battery_level = agv.battery
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                n_free_amr = state.agv_manager.get_n_free_agvs()
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                # amr_service = 1 if clipped_queue > 0 and action == 0 else 0
                # depletion_penalty = -1 if agv_battery_level <= 20 else 0
                # return amr_service - depletion_penalty - clipped_queue / 350
                if total_queue > 0 and n_free_amr == 0 and n_depleted_amr > 0:
                    return -1
                if total_queue == 0 and action > 0:
                    return 1
                else:
                    return 0

            # elif self.reward_setting == 13:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     clipped_queue = min(350, total_queue)
            #     clipped_st = min(3600, state.trackers.average_service_time)
            #     return - clipped_queue - clipped_st

            elif self.reward_setting == 14:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                # clipped_queue = min(350, total_queue)
                # clipped_st = min(3600, state.trackers.average_service_time)
                charging_util = state.agv_manager.get_average_charging_utilization()
                queue_greater_zero = 1 if total_queue > 0 else 0
                return - (charging_util / state.time) * queue_greater_zero

            elif self.reward_setting == 15:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                return - (clipped_queue / 350) * 10 #+ (state.agv_manager.get_average_agv_battery_working() / 100) * 10
                # elif self.reward_setting == 15:
            #     agv = state.agv_manager.agv_index[state.previous_agv]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     self.f_get_avg_battery(state)
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_greater_zero = 1 if total_queue > 0 else 0
            #
            #     agvm = state.agv_manager
            #     cs_queue = agvm.queued_charging_events
            #     cs = agvm.get_charging_station(agv.position)
            #     pot_start_time = state.time
            #     if cs_queue[cs]:
            #         pot_start_time = cs_queue[cs][-1].time
            #     pot_finish_time = pot_start_time + charging_duration
            #     total_time = pot_finish_time - state.time
            #     if action == 0:
            #         return 0
            #     return (charging_duration / total_time) * queue_greater_zero
            elif self.reward_setting == 16:
                clipped_st = min(14000, state.trackers.average_service_time)
                return - (clipped_st / 14000) * 10

            elif self.reward_setting == 17:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                clipped_st = min(3600, state.trackers.average_service_time)
                return - (clipped_queue / 350)  # * 10 - (clipped_st / 3600) * 10

            # elif self.reward_setting == 18:
            #     st = state.trackers.average_service_time
            #     return - st
            elif self.reward_setting == 18:
                if state.agv_manager.get_average_agv_battery() > 0.6:
                    return -1
                elif state.agv_manager.get_average_agv_battery() < 0.4:
                    return -1
                else:
                    return 1

            elif self.reward_setting == 19:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                queue_zero = 1 if total_queue == 0 else 0
                return - (clipped_queue / 350) + (action / len(state.params.charging_thresholds)) * queue_zero
            # elif self.reward_setting == 16:
                # total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                # clipped_queue = min(350, total_queue)
                # clipped_st = min(3600, state.trackers.average_service_time)
                # n_free_amr = state.agv_manager.get_n_free_agvs()
                # n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                # queue_zero = 1 if total_queue == 0 else 0
                # action_greater_zero = 1 if action > 0 else 0
                # return (- (clipped_st / 3600)
                #         - (clipped_queue / 350)
                #         # - (n_depleted_amr / state.agv_manager.n_agvs) * queue_greater_zero
                #         + action_greater_zero * queue_zero
                #         )
            elif self.reward_setting == 20:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                busy_free_agv = state.agv_manager.get_n_busy_agvs() + state.agv_manager.get_n_free_agvs()
                if total_queue > 0:
                    return busy_free_agv / state.agv_manager.n_agvs
                elif total_queue == 0:
                    return state.agv_manager.get_average_agv_battery() / 100

            elif self.reward_setting == 21:
                return - state.trackers.travel_event_statistics.average_travel_distance()

            elif self.reward_setting == 22:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                return - (clipped_queue / 350) * 10 - state.agv_manager.get_n_depleted_agvs() * 10

            elif self.reward_setting == 23:
                agv = state.agv_manager.agv_index[state.previous_agv]
                agv_battery_level = agv.battery
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(350, total_queue)
                agvm = state.agv_manager
                cs_queue = agvm.queued_charging_events
                cs = agvm.get_charging_station(agv.position)
                d_to_cs = state.agv_manager.router.get_distance(agv.position, cs)
                t_to_cs = (d_to_cs * state.agv_manager.router.unit_distance
                            / state.agv_manager.router.speed)
                pot_start_time = state.time + t_to_cs
                if cs_queue[cs]:
                    pot_start_time = cs_queue[cs][-1].time + t_to_cs
                queue_greater_zero = 1 if total_queue > 0 else 0
                waiting_time = pot_start_time - state.time
                unused = waiting_time * queue_greater_zero #clipped_queue / 350
                unused_capa = unused * state.params.consumption_rate_unloaded
                n_free_agv = state.agv_manager.get_n_free_agvs()
                n_busy_agv = state.agv_manager.get_n_busy_agvs()
                if action > 0:
                    return - clipped_queue - unused_capa + (state.params.charging_thresholds[action] - agv_battery_level) + (n_free_agv + n_busy_agv) * queue_greater_zero
                else:
                    return - clipped_queue + (agv_battery_level * clipped_queue / 350)

            # elif self.reward_setting == 24:
            #     agv = state.agv_manager.agv_index[state.current_agv]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     self.f_get_avg_battery(state)
            #     # charging_duration = to_charge / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     clipped_queue = min(350, total_queue)
            #     # travel_time = (state.params.charging_thresholds[action] - agv_battery_level) / state.params.consumption_rate_unloaded
            #     travel_left = agv.battery / state.params.consumption_rate_unloaded
            #     agvm = state.agv_manager
            #     cs_queue = agvm.queued_charging_events
            #     cs = agvm.get_charging_station(agv.position)
            #     d_to_cs = state.agv_manager.router.get_distance(agv.position, cs)
            #     t_to_cs = (d_to_cs * state.agv_manager.router.unit_distance
            #                / state.agv_manager.router.speed)
            #     pot_start_time = state.time + t_to_cs
            #     if cs_queue[cs]:
            #         pot_start_time = cs_queue[cs][-1].time + t_to_cs
            #     # pot_finish_time = pot_start_time + charging_duration
            #     # replenished = pot_finish_time - state.time
            #     queue_greater_zero = 1 if total_queue > 0 else 0
            #     # replenished = (state.params.charging_thresholds[action]
            #     #                - agv_battery_level) / state.params.consumption_rate_unloaded
            #     waiting_time = pot_start_time - state.time
            #     # unused = (travel_left - waiting_time) * queue_greater_zero
            #     unused = waiting_time * queue_greater_zero
            #     unused_capa = unused * state.params.consumption_rate_unloaded
            #     # travel_max = 100 / state.params.consumption_rate_unloaded
            #
            #     if action > 0:
            #         return - unused_capa + (state.params.charging_thresholds[action] - agv_battery_level)
            #     else:
            #         return agv_battery_level * (clipped_queue / 350) # queue_greater_zero
            #         # return agv.orders_since_last_charge

            elif self.reward_setting == 25:
                agv = state.agv_manager.agv_index[state.current_agv]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                self.f_get_avg_battery(state)
                # charging_duration = to_charge / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                # travel_time = (state.params.charging_thresholds[action] - agv_battery_level) / state.params.consumption_rate_unloaded
                travel_left = agv.battery / state.params.consumption_rate_unloaded
                agvm = state.agv_manager
                cs_queue = agvm.queued_charging_events
                cs = agvm.get_charging_station(agv.position)
                d_to_cs = state.agv_manager.router.get_distance(agv.position, cs)
                t_to_cs = (d_to_cs * state.agv_manager.router.unit_distance
                           / state.agv_manager.router.speed)
                pot_start_time = state.time + t_to_cs
                if cs_queue[cs]:
                    pot_start_time = cs_queue[cs][-1].time + t_to_cs
                # pot_finish_time = pot_start_time + charging_duration
                # replenished = pot_finish_time - state.time
                queue_greater_zero = 1 if total_queue > 0 else 0
                # replenished = (state.params.charging_thresholds[action]
                #                - agv_battery_level) / state.params.consumption_rate_unloaded
                waiting_time = pot_start_time - state.time
                # unused = (travel_left - waiting_time) * queue_greater_zero
                unused = waiting_time * queue_greater_zero
                unused_capa = unused * state.params.consumption_rate_unloaded
                # travel_max = 100 / state.params.consumption_rate_unloaded
                clipped_queue = min(350, total_queue)
                good_decision = 0
                utility = 0
                if action > 0:
                    utility = - unused_capa + state.params.charging_thresholds[action] - agv_battery_level
                if utility > 0:
                    good_decision += 1
                elif utility < 0:
                    good_decision -= 1
                return - (clipped_queue / 350) + good_decision # + (agv_battery_level / 100) * (clipped_queue / 350)

            elif self.reward_setting == 26:
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                agv_battery = agv.battery
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_queue = min(total_queue, 350)
                if state.next_e:
                    t_next_event = state.next_e.time - state.time
                else:
                    t_next_event = 0
                agvm = state.agv_manager
                free_agvs = state.agv_manager.get_n_free_agvs()
                cs_queue = agvm.queued_charging_events
                cs = agvm.get_charging_station(agv.position)
                d_to_cs = state.agv_manager.router.get_distance(agv.position, cs)
                t_to_cs = (d_to_cs * state.agv_manager.router.unit_distance
                           / state.agv_manager.router.speed)
                pot_start_time = state.time + t_to_cs
                if cs_queue[cs]:
                    pot_start_time = cs_queue[cs][-1].time + t_to_cs
                queue_greater_zero = 1 if total_queue > 0 else 0
                waiting_time = pot_start_time - state.time
                charging_time = (state.params.charging_thresholds[action] - agv_battery) / state.params.charging_rate
                positive = t_next_event - waiting_time - charging_time
                return positive
            # elif self.reward_setting == 25:
            #     agv = state.current_agv
            #     dist_to_next_order = hypot(state.I_O_positions[state.next_e.source][0] - agv.position[0],
            #           state.I_O_positions[state.next_e.source][1] - agv.position[1])
            #
            #     time_to_next_order = (dist_to_next_order * state.agv_manager.router.unit_distance
            #                           / state.agv_manager.router.speed)
            #     depl_next_order = time_to_next_order / state.params.consumption_rate_unloaded



            # elif self.reward_setting == 10:
            #     current_service_time = state.trackers.average_service_time
            #     self.service_time_last_step = current_service_time
            #     self.service_time_history.append(current_service_time)
            #
            #     # Only keep trend_window entries
            #     if len(self.service_time_history) > self.trend_window:
            #         self.service_time_history.pop(0)
            #
            #     if len(self.service_time_history) < self.trend_window:
            #         return 0
            #     else:
            #         x = np.arange(len(self.service_time_history))
            #         y = np.array(self.service_time_history)
            #         slope, _ = np.polyfit(x, y, 1)
            #
            #         # Normalize by mean and invert (negative slope = improvement = positive reward)
            #         normalized_trend = -slope / np.mean(y)
            #         return normalized_trend

            # elif self.reward_setting == 11:
            #     if self.n_observations == 0:
            #         self.n_observations += 1
            #         self.service_time_last_step = state.trackers.average_service_time
            #         return 0
            #     self.n_observations += 1
            #     current_service_time = state.trackers.average_service_time
            #     delta_service_time = self.service_time_last_step - current_service_time
            #     self.service_time_last_step = current_service_time
            #     self.running_avg += delta_service_time
            #     state.running_avg = self.running_avg
            #     self.rewards.append(delta_service_time)
            #     state.rewards = self.rewards
            #     # self.n_orders_last_step = n_orders_now
            #     return delta_service_time

            # elif self.reward_setting == 12:
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     self.f_get_avg_battery(state)
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 60 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_ratio = total_queue / 600
            #     action_ratio = charging_duration / max_charging_duration
                # if action == 0:
                #     return 1 + queue_ratio
                # return (queue_ratio * action_ratio) - queue_ratio
                # return queue_ratio * action_ratio
                # penalty_for_high_queue = -2 * queue_ratio  # Increases penalty as queue grows
                # efficiency_penalty = action_ratio * queue_ratio  # Lower penalty for efficient charging at low queues
                # if action == 0:
                #     return - queue_ratio
                # if total_queue == 0:
                #     return charging_duration
                # return penalty_for_high_queue + efficiency_penalty
                # return - (charging_duration / total_queue) - queue_ratio
                # return action_ratio / queue_ratio
                # def sigmoid_transform(x, k=10):
                #     return 2 / (1 + np.exp(k * x)) - 1  # Maps to [-1, 1]
                #
                # if action == 0:
                #     return sigmoid_transform(1 - queue_ratio)  # High queues -> good reward
                # else:
                #     # Combine queue and action ratios to reward low charging at high queues
                #     return sigmoid_transform(action_ratio * queue_ratio)
                # avg_service_time = state.trackers.average_service_time
                # working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                # if total_queue == 0:
                #     # variante 1: durch sevice time, variante 2: nur wc und cd, variante 3: mit max
                #     if action == 0:
                #         # When no orders, reward maintaining current working capacity
                #         return working_capacity #/ avg_service_time
                #     else:
                #         # When charging with no orders, consider only the opportunity cost
                #         # relative to potential future orders
                #         return -charging_duration #/ avg_service_time
                #
                # if action == 0:
                #     return working_capacity / (total_queue * avg_service_time)
                # capa that could be provided to the system but is lost due to charging
                # if working_capacity > charging_duration:
                #     opportunity_cost = - (charging_duration / (total_queue * state.trackers.average_service_time))
                # else:
                #     opportunity_cost = - (working_capacity / (total_queue * state.trackers.average_service_time))
                # opportunity_cost = - (charging_duration / (total_queue * state.trackers.average_service_time))
                # return opportunity_cost

            # elif self.reward_setting == 13:
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 60 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_ratio = total_queue / 600
            #     action_ratio = charging_duration / max_charging_duration
            #     if action == 0:
            #         return queue_ratio
            #     return (queue_ratio * action_ratio) - queue_ratio

            # elif self.reward_setting == 14:
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 80 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_ratio = total_queue / 330
            #     action_ratio = charging_duration / max_charging_duration
            #     return - queue_ratio
                # if action == 0:
                #     return 1 + queue_ratio
                # # return (queue_ratio * action_ratio) - queue_ratio
                # # return queue_ratio * action_ratio
                # penalty_for_high_queue = -1 * queue_ratio  # Increases penalty as queue grows
                # efficiency_penalty = action_ratio * (
                #         1 - queue_ratio)  # Lower penalty for efficient charging at low queues
                #
                # return penalty_for_high_queue + efficiency_penalty

            # elif self.reward_setting == 15:
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 60 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_ratio = total_queue / 600
            #     action_ratio = charging_duration / max_charging_duration
                # if action == 0:
                #     return 1 + queue_ratio
                # return (queue_ratio * action_ratio) - queue_ratio
                # return queue_ratio * action_ratio
                # penalty_for_high_queue = -2 * queue_ratio  # Increases penalty as queue grows
                # efficiency_penalty = action_ratio * queue_ratio  # Lower penalty for efficient charging at low queues
                # if action == 0:
                #     return - queue_ratio
                # if total_queue == 0:
                #     return charging_duration
                # return penalty_for_high_queue + efficiency_penalty
                # return - (charging_duration / total_queue) - queue_ratio
                # return action_ratio / queue_ratio
                # def sigmoid_transform(x, k=10):
                #     return 2 / (1 + np.exp(k * x)) - 1  # Maps to [-1, 1]
                #
                # if action == 0:
                #     return sigmoid_transform(1 - queue_ratio)  # High queues -> good reward
                # else:
                #     # Combine queue and action ratios to reward low charging at high queues
                #     return sigmoid_transform(action_ratio * queue_ratio)
            #     avg_service_time = state.trackers.average_service_time
            #     max_capacity = 80 / state.params.consumption_rate_unloaded
            #     working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
            #     if total_queue == 0:
            #         if action == 0:
            #             # When no orders, reward having high work capacity
            #             return working_capacity / max_capacity  # avg_service_time
            #         else:
            #             # When charging with no orders, reward
            #             return - charging_duration / max_charging_duration  # avg_service_time
            #
            #     if action == 0:
            #         return working_capacity / (total_queue * avg_service_time)
            #     # capa that could be provided to the system but is lost due to charging
            #     # if working_capacity > charging_duration:
            #     #     opportunity_cost = - (charging_duration / (total_queue * state.trackers.average_service_time))
            #     # else:
            #     #     opportunity_cost = - (working_capacity / (total_queue * state.trackers.average_service_time))
            #     current_working_battery = 0
            #     for agv_id, agv in state.agv_manager.agv_index.items():
            #         current_working_battery += agv.battery - 20
            #     current_working_capa = current_working_battery / (
            #             (state.params.consumption_rate_unloaded +
            #              state.params.consumption_rate_unloaded) / 2)
            #     capacity_handled = current_working_capa / (total_queue * avg_service_time)
            #     capacity_not_handled = capacity_handled * (total_queue * avg_service_time)
            #
            #     opportunity_cost = - (charging_duration / (total_queue * avg_service_time))
            #     # opportunity_cost = - (charging_duration / capacity_not_handled)
            #     return opportunity_cost
            # elif self.reward_setting == 16:
            #     current_service_time = state.trackers.average_service_time
            #     self.service_time_history.append(current_service_time)
            #
            #     # Only keep trend_window entries
            #     if len(self.service_time_history) > self.trend_window:
            #         self.service_time_history.pop(0)
            #
            #     if len(self.service_time_history) < self.trend_window:
            #         return 0
            #     else:
            #         x = np.arange(len(self.service_time_history))
            #         y = np.array(self.service_time_history)
            #         slope, _ = np.polyfit(x, y, 1)
            #         mean_service_time = np.mean(y)
            #         # Normalize by mean and invert (negative slope = improvement = positive reward)
            #         normalized_trend = -slope / mean_service_time
            #         level_reward = np.exp(-mean_service_time / 300) #- 1
            #         return normalized_trend + level_reward

            # elif self.reward_setting == 17:
            #     current_service_time = state.trackers.average_service_time
            #     self.service_time_history.append(current_service_time)
            #
            #     # Only keep trend_window entries
            #     if len(self.service_time_history) > self.trend_window:
            #         self.service_time_history.pop(0)
            #
            #
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 60 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_ratio = total_queue / 600
            #     action_ratio = charging_duration / max_charging_duration
            #     normalized_trend = 0
            #
            #     # return (queue_ratio * action_ratio) - queue_ratio
            #     # return queue_ratio * action_ratio
            #     penalty_for_high_queue = -2 * queue_ratio  # Increases penalty as queue grows
            #     efficiency_penalty = action_ratio * queue_ratio
            #     if len(self.service_time_history) < self.trend_window:
            #         return 0
            #     else:
            #         x = np.arange(len(self.service_time_history))
            #         y = np.array(self.service_time_history)
            #         slope, _ = np.polyfit(x, y, 1)
            #
            #         # Normalize by mean and invert (negative slope = improvement = positive reward)
            #         normalized_trend = -slope #/ np.mean(y)
            #     if action == 0:
            #         return 1 + queue_ratio + normalized_trend
            #     return penalty_for_high_queue + efficiency_penalty + normalized_trend

            # elif self.reward_setting == 18:
            #     current_service_time = state.trackers.average_service_time
            #     if state.agv_manager.agv_trackers.n_charges == 0:
            #         self.service_time_last_step = state.trackers.average_service_time
            #         return 0
            #     delta_service_time = self.service_time_last_step - current_service_time
            #     self.service_time_last_step = current_service_time
            #     self.running_avg += delta_service_time
            #     state.running_avg = self.running_avg
            #     self.rewards.append(delta_service_time)
            #     state.rewards = self.rewards
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 60 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     queue_ratio = total_queue / 600
            #     action_ratio = charging_duration / max_charging_duration
            #     avg_service_time = state.trackers.average_service_time
            #     working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
            #     if total_queue == 0:
            #         # reward charging if battery is low,
            #         # reward not charging if
            #         if action == 0:
            #             charging_value = 0
            #         else:
            #             charging_value = charging_duration / max_charging_duration
            #     else:
            #         # reward charging at higher queues
            #         if action > 0:
            #             charging_value = action_ratio * queue_ratio
            #         else:
            #             charging_value = - queue_ratio
            #     return charging_value + delta_service_time
            #
            # elif self.reward_setting == 19:
            #     # pallette in avg travel time + material handling -> lower bound
            #     current_service_time = state.trackers.average_service_time
            #     if state.agv_manager.agv_trackers.n_charges == 0:
            #         self.service_time_last_step = state.trackers.average_service_time
            #         return 0
            #     delta_service_time = self.service_time_last_step - current_service_time
            #     self.service_time_last_step = current_service_time
            #     self.running_avg += delta_service_time
            #     state.running_avg = self.running_avg
            #     self.rewards.append(delta_service_time)
            #     state.rewards = self.rewards
            #     agv_id = state.current_agv
            #     if agv_id is None:
            #         return 0
            #
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     max_charging_duration = 60 / state.agv_manager.charging_rate
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     avg_service_time = state.trackers.average_service_time
            #     working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
            #     # TODO total_queue == 0 raus, nivilieren
            #     if total_queue == 0:
            #         # reward charging if battery is low,
            #         # if action == 0:
            #         #     charging_value = (agv.battery / 20) / 5
            #         # else:
            #         charging_value = charging_duration / max_charging_duration
            #     else:
            #         # reward having high capacity during high queues
            #         charging_value = working_capacity / (total_queue * avg_service_time)
            #     return delta_service_time + charging_value
            #
            # elif self.reward_setting == 20:
            #     reward = 0
            #     free_cs = self.f_get_free_cs_available(state)
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     action_threshold = state.params.charging_thresholds[action]
            #     agv = state.agv_manager.agv_index[state.previous_agv]
            #     # if agv.battery <= 20:
            #     #     reward -= 1
            #     charging_ratio = (action_threshold / 100) - (agv.battery / 100)
            #
            #     if total_queue == 0 or free_cs:
            #         reward += 1
            #         # if action != 0:
            #         #     reward += 0.5
            #     # else:
            #     #     reward -= charging_ratio
            #     # if state.agv_manager.n_free_agvs == 0:
            #     #     reward -= charging_ratio
            #     n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
            #     return reward  # - n_charging_agvs_ratio - total_queue / 350
            #
            # elif self.reward_setting == 21:
            #     reward = 0
            #     free_cs = self.f_get_free_cs_available(state)
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     action_threshold = state.params.charging_thresholds[action]
            #     charging_ratio = action_threshold / 100
            #     if total_queue == 0:
            #         reward += charging_ratio
            #     if free_cs:
            #         reward += charging_ratio  # Reward proportional to charge amount
            #     else:
            #         reward -= charging_ratio
            #     if state.agv_manager.n_free_agvs == 0:
            #         reward -= charging_ratio
            #     n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
            #     return reward - n_charging_agvs_ratio - (total_queue / 350)
            #
            # elif self.reward_setting == 22:
            #     charging_duration = action
            #     agv_id = state.current_agv
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     if total_queue == 0:
            #         total_queue = 1
            #     if agv_id is None:
            #         return 0
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery
            #     to_charge = state.params.charging_thresholds[action] - agv_battery_level
            #     charging_duration = to_charge / state.agv_manager.charging_rate
            #     agvm = state.agv_manager
            #     cs_queue = agvm.queued_charging_events
            #     cs = agvm.get_charging_station(agv.position)
            #     pot_start_time = state.time
            #     if cs_queue[cs]:
            #         pot_start_time = cs_queue[cs][-1].time
            #     pot_finish_time = pot_start_time + charging_duration
            #     t_out_of_system = pot_finish_time - state.time
            #     curr_avg_st = state.trackers.average_service_time
            #     n_free_agv_ratio = state.agv_manager.n_free_agvs / state.agv_manager.n_agvs
            #     return - t_out_of_system # / (curr_avg_st * total_queue) # + n_free_agv_ratio
            #
            # elif self.reward_setting == 23:
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
            #     return - n_charging_agvs_ratio - (total_queue / 350)

            elif self.reward_setting == 24:
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
                n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
                state.last_amr_ratio = - n_charging_agvs_ratio
                state.last_queue_ratio = - (total_queue / 350)
                return reward - (total_queue / 350) - n_charging_agvs_ratio # - (clipped_st / 3600)

            # elif self.reward_setting == 25:
            #     reward = 0
            #     free_cs = self.f_get_free_cs_available(state)
            #     total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
            #     agv_id = state.current_agv
            #     agv = state.agv_manager.agv_index[agv_id]
            #     agv_battery_level = agv.battery / 100
            #
            #     if total_queue == 0 and action == 1:
            #         # prefer low battery
            #         reward += 1 + (1 - agv_battery_level)
            #     if total_queue > state.agv_manager.n_free_agvs and action == 1:
            #         reward -= 1 + agv_battery_level / self.f_get_avg_battery(state)
            #     if total_queue > state.agv_manager.n_free_agvs and action == 0:
            #         reward += 1 + agv_battery_level / self.f_get_avg_battery(state)
            #     if free_cs and action == 1:
            #         reward += 1 + agv_battery_level / self.f_get_avg_battery(state)
            #     elif not free_cs and action == 1:
            #         reward -= 1 + agv_battery_level / self.f_get_avg_battery(state)
            #     n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
            #     return reward # - (total_queue / 350) - n_charging_agvs_ratio


            # elif self.reward_setting == 26:
            #     working_battery = state.agv_manager.get_average_agv_battery_working() / 100
            #     free_battery = state.agv_manager.get_average_agv_battery_free() / 100
            #     try:
            #         assert working_battery >= 0
            #     except:
            #         print(f"reward error: {working_battery}")
            #     return working_battery + free_battery

            elif self.reward_setting == 27:
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                battery_to_charge = 0
                # return - agv.util_since_last_charge / agv.orders_since_last_charge
                if action > 0:
                    queue_len = 0
                    emergency_depletion = 0
                    if agv.battery <= 20:
                        emergency_depletion = 1
                    battery_to_charge += 100 - agv.battery
                    traveling_queue = state.agv_manager.travel_to_charging_stations
                    cs = None
                    for cs in traveling_queue.keys():
                        battery_to_charge_cs = 0
                        traveling_agvs = traveling_queue[cs]
                        for t_agv in traveling_agvs:
                            if t_agv == agv:
                                break
                            battery_to_charge_cs += 100 - t_agv.battery
                        battery_to_charge += battery_to_charge_cs
                    if cs in state.agv_manager.booked_charging_stations.keys():
                        queue = state.agv_manager.booked_charging_stations[cs]
                        queue_len = len(queue)
                        for i, a in enumerate(queue):
                            if i == 0:
                                if state.agv_manager.queued_charging_events[cs]:
                                    charging_event = state.agv_manager.queued_charging_events[cs][0]
                                    battery_to_charge += 100 - charging_event.check_battery_charge(state)
                            else:
                                battery_to_charge += 100 - a.battery
                    estimated_waiting = battery_to_charge / state.agv_manager.charging_rate
                    # max_waiting = (80 * state.agv_manager.n_agvs) / state.agv_manager.charging_rate
                    max_waiting = 100 - agv.battery / state.agv_manager.charging_rate
                    # return - estimated_waiting / max_waiting - (total_queue / 330) - emergency_depletion
                    # return - estimated_waiting / state.trackers.average_service_time - emergency_depletion - queue_len
                    return - estimated_waiting / max_waiting - emergency_depletion
                else:
                    remaining_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                    max_capacity = 80 / state.params.consumption_rate_unloaded
                    # return - (total_queue / 330) + remaining_capacity / max_capacity
                    # return remaining_capacity / state.trackers.average_service_time
                    if total_queue > 0:
                        return - (total_queue / 330) + remaining_capacity / max_capacity
                    else:
                        return - 3

            elif self.reward_setting == 28:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                if total_queue > 0:
                    return - state.agv_manager.get_n_depleted_agvs() / total_queue
                else:
                    if action == 1:
                        return 1
                    else:
                        return -1

            elif self.reward_setting == 29:
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                traveling_queue = state.agv_manager.travel_to_charging_stations
                for cs in traveling_queue.keys():
                    traveling_agvs = traveling_queue[cs]
                    for t_agv in traveling_agvs:
                        if t_agv == agv:
                            break
                penelty = 0
                if action == 1:
                    if agv.battery <= 20:
                        penelty += 1
                n_charging_agvs_ratio = (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 330
                working_battery = state.agv_manager.get_average_agv_battery_working()
                return - penelty - n_charging_agvs_ratio

            elif self.reward_setting == 30:
                return - state.trackers.average_service_time / 160

            elif self.reward_setting == 31:
                # go charging reward
                reward = 0
                free_cs = self.f_get_free_cs_available(state)
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                # if total_queue == 0 and action == 1:
                #     reward += 1
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                if agv.battery <= 20 and action == 1:
                    reward -= 2
                if free_cs and action == 1:
                    reward += 1
                elif not free_cs and action == 1:
                    reward -= 1
                # if state.agv_manager.n_free_agvs == 0 and action == 1:
                #     reward -= 1
                n_charging_agvs_ratio = (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)

                # return reward
                # curr_avg_st = state.trackers.average_service_time
                # interrupt reward
                # cs_queue = state.agv_manager.queued_charging_events
                interrupt_reward = 0
                return reward

            elif self.reward_setting == 32:
                free_cs = self.f_get_free_cs_available(state)
                total_queue = (state.trackers.n_queued_delivery_orders +
                               state.trackers.n_queued_retrieval_orders)
                n_free_agvs = state.agv_manager.get_n_free_agvs()
                n_busy_agvs = state.agv_manager.get_n_busy_agvs()
                total_agvs = state.agv_manager.n_agvs
                n_depleted_agvs = state.agv_manager.get_n_depleted_agvs()

                # Position Metrics (like trading position management)
                charging_position = n_depleted_agvs / total_agvs
                service_capacity = (n_busy_agvs + n_free_agvs) / total_agvs
                queue_pressure = min(1.0, total_queue / 350)  # Normalized queue pressure

                if action == 1:  # CHARGE decision
                    reward = 0.0

                    # 1. Market Timing Component
                    # Like entering a position at the right time
                    if total_queue == 0:
                        timing_bonus = 1.0
                    else:
                        # Only use queue_pressure here for timing decisions
                        timing_bonus = -queue_pressure
                    reward += timing_bonus
                    reward += n_free_agvs / total_agvs

                    # 2. Resource Availability Component
                    # Like liquidity in trading
                    if free_cs:
                        resource_bonus = 1.0
                    else:
                        resource_bonus = -1.0
                    reward += resource_bonus

                    # 3. Position Sizing Penalty
                    # Like oversized position risk
                    if n_free_agvs == 0:
                        capacity_penalty = -1.0
                    else:
                        capacity_penalty = 0.0
                    reward += capacity_penalty

                    # 4. Inventory Management
                    # Like portfolio balance
                    position_penalty = -charging_position
                    reward += position_penalty

                else:  # NO CHARGE decision
                    reward = 0.0

                    # 1. Service Capacity Reward
                    # Like profit from good position
                    reward += service_capacity

                    # 2. Queue Management Reward
                    # Like market making profit
                    if total_queue > 0 and n_free_agvs > 0:
                        reward += 0.5

                    # 3. Position Management
                    # Like carrying cost of position
                    reward -= charging_position

                # Only apply queue_pressure in final risk adjustment
                # This better reflects overall market conditions
                risk_adjustment = 1.0 - (0.5 * queue_pressure)

                return reward * risk_adjustment

            elif self.reward_setting == 33:
                reward = 1
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                if action == 1 and state.orders_next_hour < 50:
                    return reward + 2
                else:
                    if state.orders_next_hour == 0:
                        return - 1 - agv.battery / 100
                    return state.agv_manager.get_n_busy_agvs() / state.orders_next_hour

            elif self.reward_setting == 34:
                reward = 2
                total_queue = (state.trackers.n_queued_delivery_orders +
                               state.trackers.n_queued_retrieval_orders)
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                if action == 1:
                    if agv.battery <= 20:
                        reward -= 0.5
                    if action == 1 and total_queue < 10:
                        reward += 1
                else:
                    if total_queue == 0:
                        reward += agv.battery / 100
                    # return reward + state.agv_manager.get_n_busy_agvs() / total_queue
                return reward - state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs

            elif self.reward_setting == 35:
                agv_id = state.previous_agv
                agv = state.agv_manager.agv_index[agv_id]
                if action == 1:
                    if state.agv_manager.get_n_depleted_agvs() > 1:
                        return -1
                    else:
                        return 1
                else:
                    if agv.battery < 20:
                        return - 5
                    return 1 - state.trackers.n_queued_retrieval_orders / 330

            elif self.reward_setting == 36:
                if state.done:
                    print("done reward")
                    return - state.trackers.average_service_time
                else:
                    agv_id = state.previous_agv
                    agv = state.agv_manager.agv_index[agv_id]
                    if agv.battery < 20:
                        return -1
                    else:
                        return 1

            elif self.reward_setting == 37:
                # go charging reward
                curr_avg_st = state.trackers.average_service_time
                agv_id = state.current_agv
                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                if agv_battery_level < 20 and action == 0:
                    return -10
                reward = 0
                free_cs = self.f_get_free_cs_available(state)
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                if total_queue == 0 and action == 1:
                    reward += 1
                # elif total_queue > 20 and action == 1:
                #     reward -= 1
                if free_cs and action == 1:
                    reward += 1
                elif not free_cs and action == 1:
                    reward -= 1
                if state.agv_manager.n_free_agvs == 0 and action == 1:
                    reward -= 1
                n_charging_agvs_ratio = (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)
                return reward - (curr_avg_st / 1000)  # - n_charging_agvs_ratio - (total_queue / 350) # - (curr_avg_st / 600)

            elif self.reward_setting == 37:
                # go charging reward
                curr_avg_st = state.trackers.average_service_time
                agv_id = state.current_agv
                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                # if agv_battery_level < 20 and action == 0:
                #     return -10
                reward = 0
                free_cs = self.f_get_free_cs_available(state)
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                if total_queue == 0 and action == 1:
                    reward += 1
                # elif total_queue > 20 and action == 1:
                #     reward -= 1
                if free_cs and action == 1:
                    reward += 1
                elif not free_cs and action == 1:
                    reward -= 1
                if state.agv_manager.n_free_agvs == 0 and action == 1:
                    reward -= 1
                n_charging_agvs_ratio = (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)
                return reward - n_charging_agvs_ratio - (total_queue / 350) # - (curr_avg_st / 1000)

            elif self.reward_setting == 38:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                agv = state.agv_manager.agv_index[state.previous_agv]
                battery = agv.battery

                if action > 0:
                    time_charging = (state.params.charging_thresholds[action] - battery) / state.params.charging_rate
                else:
                    time_charging = 0
                max_time_charging = (state.params.charging_thresholds[-1] - battery) / state.params.charging_rate
                # estimated_waiting = (time_charging * state.agv_manager.get_n_depleted_agvs()) / ((80 / state.params.charging_rate) * state.agv_manager.n_agvs)
                # estimated_waiting = time_charging / max_time_charging
                estimated_waiting = time_charging * state.agv_manager.get_n_depleted_agvs()

                # orders_next_hour = state.orders_next_hour
                # > 0 if action == 0
                time_left = battery / state.params.consumption_rate_loaded
                st_next_h = ((total_queue * state.trackers.average_service_time) /
                             state.agv_manager.n_agvs * self.f_get_n_working_agvs(state))
                if st_next_h > 0:
                    # trade_off = (time_left / st_next_h) - (estimated_waiting * 1 if action > 0 else 0)
                    trade_off = min(((time_left - estimated_waiting) / st_next_h), 1)
                else:
                    # trade_off = (estimated_waiting * 1 if action > 0 else 0)
                    trade_off = time_charging / max_time_charging
                return trade_off

            elif self.reward_setting == 39:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders

                if total_queue > 0:
                    return - total_queue / 350 - (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)
                else:
                    return state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs

            elif self.reward_setting == 40:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders

                if total_queue > 0:
                    return - total_queue / 350 - state.agv_manager.get_n_depleted_agvs()
                else:
                    return state.agv_manager.get_n_depleted_agvs()

            elif self.reward_setting == 41:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders

                if total_queue > 0:
                    return - state.agv_manager.get_n_depleted_agvs()
                else:
                    return state.agv_manager.get_n_depleted_agvs()

            elif self.reward_setting == 42:
                # total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                return - state.agv_manager.get_n_depleted_agvs()

            elif self.reward_setting == 43:
                return - state.trackers.average_service_time / 1000

            elif self.reward_setting == 44:
                total_queue = state.trackers.n_queued_retrieval_orders + state.trackers.n_queued_delivery_orders
                return - (total_queue / 350) - (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)
            elif self.reward_setting == 45:
                total_queue = state.trackers.n_queued_retrieval_orders + state.trackers.n_queued_delivery_orders
                return - (total_queue / 350)

            elif self.reward_setting == 46:
                base_reward = 1
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders

                if total_queue > 0:
                    return base_reward - (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)
                else:
                    return base_reward + (state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs)

            elif self.reward_setting == 47:
                base_reward = 1
                reward = 0
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                n_depleted_amr = state.agv_manager.get_n_depleted_agvs()
                if action > 0:
                    if total_queue > n_depleted_amr:
                        reward -= 1
                    elif total_queue < n_depleted_amr:
                        reward += 1
                    if self.f_get_free_cs_available(state):
                        reward += 1
                return base_reward + reward

            elif self.reward_setting == 48:
                # Goal: Get battery as "cheap" as possible
                # time charging * price
                # price = order queues + cs queue
                total_queue = max((state.trackers.n_queued_delivery_orders +
                                   state.trackers.n_queued_retrieval_orders),
                                  350)
                queue_ratio = total_queue / 350  # [0,1]
                agv = state.agv_manager.agv_index[state.previous_agv]
                avg_charging_ratio = state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs
                price = queue_ratio + avg_charging_ratio + self.f_get_dist_to_cs(state)
                if action > 0:
                    if price < 2:
                        return 1
                    elif price < 1:
                        return 2
                    else:
                        return - price
                else:
                    return - price

            elif self.reward_setting == 49:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                return - total_queue / 350

            elif self.reward_setting == 50:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders

                return max(-1, -total_queue / 350)

            elif self.reward_setting == 51:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_total_queue = max((total_queue, 350))
                queue_ratio = clipped_total_queue / 350
                n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
                charging_reward = 0
                if action > 0:
                    dist_to_cs = - self.f_get_dist_to_cs(state)
                    free_station = self.f_get_free_cs_available(state)
                    no_queue = 1 if total_queue == 0 else 0
                    no_free_agv_penalty = action if state.agv_manager.n_free_agvs == 0 else 0
                    no_free_cs_penalty = action if not free_station else 0
                    charging_reward = dist_to_cs + free_station + no_queue - no_free_agv_penalty - no_free_cs_penalty
                return charging_reward - queue_ratio - n_charging_agvs_ratio

            elif self.reward_setting == 52:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                clipped_total_queue = max((total_queue, 350))
                queue_ratio = clipped_total_queue / 350
                n_charging_agvs_ratio = state.agv_manager.n_charging_agvs / state.agv_manager.n_agvs
                charging_reward = 0
                if action > 0:
                    # dist_to_cs = - self.f_get_dist_to_cs(state)
                    free_station = self.f_get_free_cs_available(state)
                    no_queue = 1 if total_queue == 0 else 0
                    no_free_agv_penalty = action if state.agv_manager.n_free_agvs == 0 else 0
                    no_free_cs_penalty = action if not free_station else 0
                    charging_reward = free_station + no_queue - no_free_agv_penalty - no_free_cs_penalty
                return charging_reward - queue_ratio - n_charging_agvs_ratio

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
