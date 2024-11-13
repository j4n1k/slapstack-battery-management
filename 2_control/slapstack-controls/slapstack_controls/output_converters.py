from math import sqrt, inf

import numpy as np

from slapstack.core_state import State
from slapstack.core_state_location_manager import LocationTrackers
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
        self.time_last_step = 0
        self.n_stacks = 3
        self.running_avg = 0
        self.feature_stack = np.zeros((0, 0))
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
    def f_get_curr_agv_battery(state: State):
        agv_id = state.current_agv
        if agv_id == None:
            return 0
        else:
            agvm = state.agv_manager
            agv = agvm.agv_index[agv_id]
            assert agv.battery > 0
            # try:
            #     assert agv.battery >= 20
            # except:
            #     print(f"Error in Converter: Battery below Threshold: {agv.battery} ")
            if agv.battery == 0:
                print()
            return agv.battery / 100

    @staticmethod
    def f_get_n_depleted_agvs(state: State):
        agvm = state.agv_manager
        return agvm.get_n_depleted_agvs() / agvm.n_agvs

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
        return state.time / (24 * 3600)

    def f_get_lane_fill_level_avg(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.average(self.fill_level_per_lane)

    def f_get_lane_fill_level_std(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.std(self.fill_level_per_lane)

    def f_get_avg_entropy(self, state: State):
        lt = state.location_manager.location_trackers
        lane_entropy = lt.lane_wise_entropies.values()
        return sum(lane_entropy) / len(lane_entropy)

    @staticmethod
    def f_get_global_fill_level(state: State):
        # lt: LocationTrackers = state.location_manager.location_trackers
        # return 1 - lt.n_open_locations / lt.n_total_locations
        return state.trackers.get_fill_level()

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
        if n_ret > self.max_retrieval_buffer_len:
            self.max_retrieval_buffer_len = n_ret
        if self.max_retrieval_buffer_len == 0:
            return 0
        else:
            return (state.trackers.n_queued_retrieval_orders
                    / self.max_retrieval_buffer_len)

    def f_get_queue_len_delivery_orders(self, state: State):
        n_del = state.trackers.n_queued_delivery_orders
        if n_del > self.max_delivery_buffer_len:
            self.max_delivery_buffer_len = n_del
        if self.max_delivery_buffer_len == 0:
            return 0
        else:
            return (state.trackers.n_queued_delivery_orders
                    / self.max_delivery_buffer_len)

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
    def f_get_agv_id(state: State):
        agv_id = state.current_agv
        if not agv_id:
            return 0
        else:
            return state.current_agv / len(state.agv_manager.agv_index)

    @staticmethod
    def f_get_orders_not_served(state: State):
        pass

    @staticmethod
    def f_get_free_agv(state: State):
        return state.agv_manager.n_free_agvs / state.agv_manager.n_agvs

    def modify_state(self, state: State) -> np.ndarray:
        # if state.decision_mode == "charging":
        if state.agv_manager.agv_trackers.n_charges == 0:
            self.time_last_step = state.time
        features = []
        for feature_name in self.feature_list:
            features.append(getattr(self, f'f_get_{feature_name}')(state))
        if not np.any(self.feature_stack):
            self.feature_stack = np.zeros((self.n_stacks,
                                          len(self.feature_list)))

        self.feature_stack = np.stack((self.feature_stack[1],
                                       self.feature_stack[2],
                                       np.array(features)))
        #self.time_last_step = state.time
        #return np.array(features)
        return np.array(features)
        # return self.feature_stack
        # else:
        #     return np.zeros((self.n_stacks, len(self.feature_list)))

    def reset(self):
        self.n_observations = 0
        self.n_observations_prev = 0
        self.n_orders_last_step = 0
        self.n_delivery_orders_last_step = 0
        self.n_retrieval_orders_last_step = 0
        self.service_time_last_step = 0
        self.time_last_step = 0
        self.running_avg = 0
        self.rewards = []
        self.util_last_step = 0
        # self.service_time_history = []

    def calculate_reward(self,
                         state: State,
                         action: int,
                         legal_actions: list,
                         decision_mode: str) -> float:
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
            if self.reward_setting == 1:
                self.n_observations += 1
                current_service_time = state.trackers.average_service_time
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                self.running_avg += delta_service_time
                state.running_avg = self.running_avg
                self.rewards.append(delta_service_time)
                state.rewards = self.rewards
                # self.n_orders_last_step = n_orders_now
                return delta_service_time
            #Reward 2
            elif self.reward_setting == 2:
                if state.agv_manager.agv_trackers.n_charges == 0:
                    self.service_time_last_step = state.trackers.average_service_time
                    return 0

                current_service_time = state.trackers.average_service_time
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                # self.n_orders_last_step = n_orders_now
                return delta_service_time
            elif self.reward_setting == 3:
                return - state.trackers.average_service_time
            elif self.reward_setting == 4:
                agv_id = state.current_agv
                if not agv_id:
                    return 0
                else:
                    mean_util = state.agv_manager.get_average_utilization() / state.time
                    mean_battery = state.agv_manager.get_average_agv_battery() / 100
                    return mean_util # - mean_battery
            elif self.reward_setting == 5:
                agv_id = state.current_agv
                if not agv_id:
                    return 0
                else:
                    self.n_observations += 1
                    current_util = state.agv_manager.get_average_utilization() / state.time
                    delta_util = self.util_last_step - current_util
                    self.util_last_step = current_util
                    self.running_avg += delta_util
                    state.running_avg = self.running_avg
                    self.rewards.append(delta_util)
                    state.rewards = self.rewards
                    # self.n_orders_last_step = n_orders_now
                    return delta_util
            elif self.reward_setting == 6:
                agv_id = state.current_agv
                if not agv_id:
                    return 0
                else:
                    agv = state.agv_manager.agv_index[agv_id]
                    return agv.util_since_last_charge / state.time if state.time > 0 else 0

            elif self.reward_setting == 7:
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                n_depleted_ratio = state.agv_manager.get_n_depleted_agvs() / state.agv_manager.n_agvs

                # Use a fixed reference value instead of dynamic max
                queue_capacity = 400  # or whatever makes sense for your system
                queue_ratio = total_queue / queue_capacity

                # Inverse relationship: reward depleted AGVs more when queue is small
                if total_queue == 0:
                    # Maximum reward for charging during empty queue
                    if action == 0:
                        reward = -1
                else:
                    # Decrease reward as queue grows
                    reward = 1

                # Optionally add penalty for high queue + high depletion
                if queue_ratio > 0.3 and n_depleted_ratio > 0.3:
                    reward -= (queue_ratio * n_depleted_ratio)

                return reward

            elif self.reward_setting == 8:
                return - state.agv_manager.get_n_depleted_agvs()

            elif self.reward_setting == 9:
                current_service_time = state.trackers.average_service_time
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                self.service_time_history.append(current_service_time)

                # Only keep trend_window entries
                if len(self.service_time_history) > self.trend_window:
                    self.service_time_history.pop(0)


                if len(self.service_time_history) < self.trend_window:
                    return delta_service_time
                else:
                    x = np.arange(len(self.service_time_history))
                    y = np.array(self.service_time_history)
                    slope, _ = np.polyfit(x, y, 1)

                    # Normalize by mean and invert (negative slope = improvement = positive reward)
                    normalized_trend = -slope / np.mean(y)
                    return delta_service_time - normalized_trend
                    # return delta_service_time - slope

            elif self.reward_setting == 10:
                current_service_time = state.trackers.average_service_time
                self.service_time_last_step = current_service_time
                self.service_time_history.append(current_service_time)

                # Only keep trend_window entries
                if len(self.service_time_history) > self.trend_window:
                    self.service_time_history.pop(0)

                if len(self.service_time_history) < self.trend_window:
                    return 0
                else:
                    x = np.arange(len(self.service_time_history))
                    y = np.array(self.service_time_history)
                    slope, _ = np.polyfit(x, y, 1)

                    # Normalize by mean and invert (negative slope = improvement = positive reward)
                    normalized_trend = -slope / np.mean(y)
                    return normalized_trend

            elif self.reward_setting == 11:
                if self.n_observations == 0:
                    self.n_observations += 1
                    self.service_time_last_step = state.trackers.average_service_time
                    return 0
                self.n_observations += 1
                current_service_time = state.trackers.average_service_time
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                self.running_avg += delta_service_time
                state.running_avg = self.running_avg
                self.rewards.append(delta_service_time)
                state.rewards = self.rewards
                # self.n_orders_last_step = n_orders_now
                return delta_service_time

            elif self.reward_setting == 12:
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 600
                action_ratio = charging_duration / max_charging_duration
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
                avg_service_time = state.trackers.average_service_time
                working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                if total_queue == 0:
                    # variante 1: durch sevice time, variante 2: nur wc und cd, variante 3: mit max
                    if action == 0:
                        # When no orders, reward maintaining current working capacity
                        return working_capacity #/ avg_service_time
                    else:
                        # When charging with no orders, consider only the opportunity cost
                        # relative to potential future orders
                        return -charging_duration #/ avg_service_time

                if action == 0:
                    return working_capacity / (total_queue * avg_service_time)
                # capa that could be provided to the system but is lost due to charging
                # if working_capacity > charging_duration:
                #     opportunity_cost = - (charging_duration / (total_queue * state.trackers.average_service_time))
                # else:
                #     opportunity_cost = - (working_capacity / (total_queue * state.trackers.average_service_time))
                opportunity_cost = - (charging_duration / (total_queue * state.trackers.average_service_time))
                return opportunity_cost

            elif self.reward_setting == 13:
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 600
                action_ratio = charging_duration / max_charging_duration
                if action == 0:
                    return queue_ratio
                return (queue_ratio * action_ratio) - queue_ratio

            elif self.reward_setting == 14:
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 600
                action_ratio = charging_duration / max_charging_duration
                if action == 0:
                    return 1 + queue_ratio
                # return (queue_ratio * action_ratio) - queue_ratio
                # return queue_ratio * action_ratio
                penalty_for_high_queue = -2 * queue_ratio  # Increases penalty as queue grows
                efficiency_penalty = action_ratio * (
                        1 - queue_ratio)  # Lower penalty for efficient charging at low queues

                return penalty_for_high_queue + efficiency_penalty

            elif self.reward_setting == 15:
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 600
                action_ratio = charging_duration / max_charging_duration
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
                avg_service_time = state.trackers.average_service_time
                max_capacity = 80 / state.params.consumption_rate_unloaded
                working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                if total_queue == 0:
                    if action == 0:
                        # When no orders, reward having high work capacity
                        return working_capacity / max_capacity  # avg_service_time
                    else:
                        # When charging with no orders, reward
                        return - charging_duration / max_charging_duration  # avg_service_time

                if action == 0:
                    return working_capacity / (total_queue * avg_service_time)
                # capa that could be provided to the system but is lost due to charging
                # if working_capacity > charging_duration:
                #     opportunity_cost = - (charging_duration / (total_queue * state.trackers.average_service_time))
                # else:
                #     opportunity_cost = - (working_capacity / (total_queue * state.trackers.average_service_time))
                current_working_battery = 0
                for agv_id, agv in state.agv_manager.agv_index.items():
                    current_working_battery += agv.battery - 20
                current_working_capa = current_working_battery / (
                        (state.params.consumption_rate_unloaded +
                         state.params.consumption_rate_unloaded) / 2)
                capacity_handled = current_working_capa / (total_queue * avg_service_time)
                capacity_not_handled = capacity_handled * (total_queue * avg_service_time)

                opportunity_cost = - (charging_duration / (total_queue * avg_service_time))
                # opportunity_cost = - (charging_duration / capacity_not_handled)
                return opportunity_cost
            elif self.reward_setting == 16:
                current_service_time = state.trackers.average_service_time
                self.service_time_history.append(current_service_time)

                # Only keep trend_window entries
                if len(self.service_time_history) > self.trend_window:
                    self.service_time_history.pop(0)

                if len(self.service_time_history) < self.trend_window:
                    return 0
                else:
                    x = np.arange(len(self.service_time_history))
                    y = np.array(self.service_time_history)
                    slope, _ = np.polyfit(x, y, 1)
                    mean_service_time = np.mean(y)
                    # Normalize by mean and invert (negative slope = improvement = positive reward)
                    normalized_trend = -slope / mean_service_time
                    level_reward = np.exp(-mean_service_time / 300) #- 1
                    return normalized_trend + level_reward

            elif self.reward_setting == 17:
                current_service_time = state.trackers.average_service_time
                self.service_time_history.append(current_service_time)

                # Only keep trend_window entries
                if len(self.service_time_history) > self.trend_window:
                    self.service_time_history.pop(0)


                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 600
                action_ratio = charging_duration / max_charging_duration
                normalized_trend = 0

                # return (queue_ratio * action_ratio) - queue_ratio
                # return queue_ratio * action_ratio
                penalty_for_high_queue = -2 * queue_ratio  # Increases penalty as queue grows
                efficiency_penalty = action_ratio * queue_ratio
                if len(self.service_time_history) < self.trend_window:
                    return 0
                else:
                    x = np.arange(len(self.service_time_history))
                    y = np.array(self.service_time_history)
                    slope, _ = np.polyfit(x, y, 1)

                    # Normalize by mean and invert (negative slope = improvement = positive reward)
                    normalized_trend = -slope #/ np.mean(y)
                if action == 0:
                    return 1 + queue_ratio + normalized_trend
                return penalty_for_high_queue + efficiency_penalty + normalized_trend

            elif self.reward_setting == 18:
                current_service_time = state.trackers.average_service_time
                if state.agv_manager.agv_trackers.n_charges == 0:
                    self.service_time_last_step = state.trackers.average_service_time
                    return 0
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                self.running_avg += delta_service_time
                state.running_avg = self.running_avg
                self.rewards.append(delta_service_time)
                state.rewards = self.rewards
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                queue_ratio = total_queue / 600
                action_ratio = charging_duration / max_charging_duration
                avg_service_time = state.trackers.average_service_time
                working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                if total_queue == 0:
                    # reward charging if battery is low,
                    # reward not charging if
                    if action == 0:
                        charging_value = 0
                    else:
                        charging_value = charging_duration / max_charging_duration
                else:
                    # reward charging at higher queues
                    if action > 0:
                        charging_value = action_ratio * queue_ratio
                    else:
                        charging_value = - queue_ratio
                return charging_value + delta_service_time

            elif self.reward_setting == 19:
                # pallette in avg travel time + material handling -> lower bound
                current_service_time = state.trackers.average_service_time
                if state.agv_manager.agv_trackers.n_charges == 0:
                    self.service_time_last_step = state.trackers.average_service_time
                    return 0
                delta_service_time = self.service_time_last_step - current_service_time
                self.service_time_last_step = current_service_time
                self.running_avg += delta_service_time
                state.running_avg = self.running_avg
                self.rewards.append(delta_service_time)
                state.rewards = self.rewards
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                agv_battery_level = agv.battery
                to_charge = state.params.charging_thresholds[action] - agv_battery_level
                charging_duration = to_charge / state.agv_manager.charging_rate
                max_charging_duration = 60 / state.agv_manager.charging_rate
                total_queue = state.trackers.n_queued_delivery_orders + state.trackers.n_queued_retrieval_orders
                avg_service_time = state.trackers.average_service_time
                working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                # TODO total_queue == 0 raus, nivilieren
                if total_queue == 0:
                    # reward charging if battery is low,
                    # if action == 0:
                    #     charging_value = (agv.battery / 20) / 5
                    # else:
                    charging_value = charging_duration / max_charging_duration
                else:
                    # reward having high capacity during high queues
                    charging_value = working_capacity / (total_queue * avg_service_time)
                return delta_service_time + charging_value

            elif self.reward_setting == 20:
                agv_id = state.current_agv
                if agv_id is None:
                    return 0

                agv = state.agv_manager.agv_index[agv_id]
                working_capacity = (agv.battery - 20) / state.params.consumption_rate_unloaded
                max_capa = 80 / state.params.consumption_rate_unloaded
                charging_value = working_capacity / max_capa
                return charging_value
        else:
            return 0

    def _calculate_trend(self) -> float:
        if len(self.service_time_history) < self.trend_window:
            return 0.0

        recent_history = self.service_time_history[-self.trend_window:]
        x = np.arange(len(recent_history))
        y = np.array(recent_history)

        # Calculate slope using numpy's polyfit
        slope, _ = np.polyfit(x, y, 1)

        # Normalize slope and invert (negative slope = improvement = positive reward)
        return -slope / np.mean(recent_history)

    @staticmethod
    def valid_action_mask(self, state: State):
        agv_id = state.current_agv
        agv = state.agv_manager.agv_index[agv_id]
        battery_level = agv.battery
        charging_thresholds = np.array(state.params.charging_thresholds)
        mask = (charging_thresholds == 0) | (charging_thresholds > battery_level)
        return mask