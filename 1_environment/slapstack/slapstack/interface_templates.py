import json
from collections import defaultdict
from os.path import join, abspath, sep
from typing import Tuple, TYPE_CHECKING, Union, Dict, Set, List

from numpy import genfromtxt

from slapstack.helpers import StorageKeys

import numpy as np


if TYPE_CHECKING:
    from slapstack.core_state import State


class SlapLogger:
    def __init__(self, dirpath: str):
        self.log_dir = dirpath
        self.slap_state = None

    def set_state(self, state: 'State'):
        self.slap_state = state

    def log_state(self):
        raise NotImplementedError


class StorageStrategy:
    def __init__(self, strategy_type):
        assert strategy_type in ['retrieval', 'delivery']
        self.type = strategy_type

    def get_action(self, state: 'State') -> Tuple[int, int, int]:
        pass

    def update(self, action: Tuple[int, int, int]):
        pass


class ChargingStrategy:
    def __int__(self, strategy_type):
        self.type = strategy_type

    def get_action(self, state: 'State', agv_id: int) -> int:
        pass

    def update(self, action: Tuple[int, int, int]):
        pass


class OutputConverter:
    def __init__(self, feature_list):
        self.feature_list = feature_list
    def modify_state(self, state: 'State') -> np.ndarray:
        pass

    def calculate_reward(self, state: 'State', action: int,
                         legal_actions: list, decision_mode: str):
        pass

    def reset(self):
        pass

class SimulationParameters:
    def __init__(self,
                 n_agvs: int,
                 pure_lanes: bool,
                 generate_orders: bool,
                 initial_pallets_storage_strategy: StorageStrategy,
                 resetting,
                 desired_fill_level: float = None,
                 n_skus: int = None,
                 all_skus: Set[int] = None,
                 n_orders: Union[int, None] = None,
                 order_list=None,
                 initial_pallets_sku_counts: Dict[int, int] = None,
                 verbose: bool = False,
                 agv_speed: float = 2.0,
                 unit_distance: float = 1.4,
                 pallet_shift_penalty_factor: int = 10,
                 n_rows: int = None,
                 n_columns: int = None,
                 n_levels: int = None,
                 n_sources: int = None,
                 n_sinks: int = None,
                 layout: Union[None, np.ndarray] = None,
                 start_period: int = None,
                 n_sku_in: Union[None, Dict[int, Dict[int, int]]] = None,
                 n_sku_out: Union[None, Dict[int, Dict[int, int]]] = None,
                 compute_feature_trackers: bool = True,
                 use_case: str = None,
                 use_case_n_partitions: int = None,
                 use_case_partition_to_use: int = None,
                 door_to_door: bool = False,
                 agv_forks: int = 1,
                 material_handling_time: int = 15,
                 update_partial_paths: bool = False,
                 battery_capacity: float = 80,  # 52 80
                 battery_consumption_h: float = 10,  # 10
                 battery_consumption_loaded_h: float = 20,  # 15
                 battery_charging_h: float = 80,  # 80
                 charging_thresholds: Union[list[int], Tuple[float, float]] = None,
                 ):

        # The inpt that are not required when usecase is provided.
        # https://www.kuka.com/en-de/products/mobility/mobile-platforms/kmp-1500
        self.battery_charging_h = battery_charging_h
        self.battery_consumption_h = battery_consumption_h
        self.battery_consumption_loaded_h = battery_consumption_loaded_h
        self.battery_capacity = battery_capacity
        self.consumption_rate_unloaded = ((self.battery_consumption_h / 3600)
                                          / self.battery_capacity) * 100
        self.consumption_rate_loaded = ((self.battery_consumption_loaded_h /
                                         3600) / self.battery_capacity) * 100
        self.charging_rate = ((self.battery_charging_h / 3600)
                              / self.battery_capacity) * 100
        self.charging_thresholds = charging_thresholds
        self.material_handling_time = material_handling_time
        optionals = [
            n_rows, n_columns, n_levels, n_skus, all_skus, n_orders, order_list,
            initial_pallets_sku_counts, n_sources, n_sinks, layout, n_sku_in,
            n_sku_out, start_period, desired_fill_level
        ]

        if use_case is not None:
            assert use_case_partition_to_use is not None
            assert use_case_n_partitions is not None
            assert use_case_partition_to_use <= use_case_n_partitions
            self.n_levels = n_levels
            self.use_case_name = use_case
            self.use_case_n_partitions = use_case_n_partitions
            self.use_case_partition_to_use = use_case_partition_to_use
            self.layout_path, self.order_path, self.initial_sku_path = (
                SimulationParameters.setup_paths(use_case))
            if use_case_n_partitions > 1:
                self.get_initial_partitions_data(use_case_n_partitions)
            use_case_partitions = self.partition_use_case(use_case_n_partitions)
            use_case = use_case_partitions[0]
            self.n_rows = use_case.layout.shape[0]
            self.n_columns = use_case.layout.shape[1]
            self.n_skus = len(use_case.distinct_skus)
            self.all_skus = use_case.distinct_skus
            self.n_orders = len(use_case.order_list)
            self.order_list = use_case.order_list
            # self.initial_pallets_sku_counts = use_case.initial_skus[1]
            partition_idx = None
            if self.use_case_n_partitions > 1:
                partition_idx = self.use_case_partition_to_use
            self.initial_pallets_sku_counts, _ = self.load_initial_skus(partition_idx)
            self.n_sources = len(
                np.argwhere(use_case.layout == StorageKeys.SOURCE))
            self.n_sinks = len(np.argwhere(use_case.layout == StorageKeys.SINK))
            self.layout = use_case.layout
            self.n_skus_in = use_case.sku_in_counts
            self.n_skus_out = use_case.sku_out_counts
            self.sku_period = use_case.current_week
            self.desired_fill_level = None
            self.shape = use_case.layout.shape + (use_case.n_levels,)
            self.door_to_door = door_to_door
            self.n_forks = agv_forks
            self.update_partial_paths = update_partial_paths
            if any(optionals):
                print("WARNING: some of the following options were passed to "
                      "SimulationParameters, but have been overridden due to "
                      "the passed usecase:")
                print(
                    "\tn_rows\n"
                    "\tn_columns\n"
                    "\tn_levels\n"
                    "\tn_skus\n"
                    "\tall_skus\n"
                    "\tn_orders\n"
                    "\torder_list\n"
                    "\tinitial_pallets_sku_counts\n"
                    "\tn_sources\n"
                    "\tn_sinks\n"
                    "\tlayout\n"
                    "\tn_sku_in\n"
                    "\tn_sku_out\n"
                    "\tstart_period\n"
                    "\tdesired_fill_level\n"
                )
        # elif all(optionals):
        else:
            self.n_rows = n_rows
            self.n_columns = n_columns
            self.n_levels = n_levels
            self.n_skus = n_skus
            self.all_skus = all_skus
            self.n_orders = n_orders
            self.order_list = order_list
            self.initial_pallets_sku_counts = initial_pallets_sku_counts
            self.n_sources = n_sources
            self.n_sinks = n_sinks
            self.layout = layout
            self.n_sku_in = n_sku_in
            self.n_sku_out = n_sku_out
            self.sku_period = start_period
            self.desired_fill_level = desired_fill_level
            self.shape = (n_rows, n_columns, n_levels)

        self.n_agvs = n_agvs
        self.pure_lanes = pure_lanes
        self.generate_orders = generate_orders
        self.initial_pallets_storage_strategy = initial_pallets_storage_strategy
        self.resetting = resetting
        self.verbose = verbose
        self.agv_speed = agv_speed
        self.unit_distance = unit_distance
        self.shift_penalty = pallet_shift_penalty_factor
        self.compute_feature_trackers = compute_feature_trackers

    def partition_use_case(self, n_partitions=40, partition_idx=None):
        """
        Splits the use case orders into n_partitions equal sections.

        :return: The use case partitions.
        """
        if self.use_case_n_partitions > 1:
            if not partition_idx:
                partition_idx = self.use_case_partition_to_use
        order_data = self.load_orders(partition_idx)
        all_skus = SimulationParameters.get_unique_skus(order_data)
        # skus_ini = get_initial_skus(order_data,
        #                             warehouse_capacity=19000,
        #                             initial_fill_level=0.6,
        #                             percent_nowait=0.8)
        skus_ini, _ = self.load_initial_skus(partition_idx)
        part_size = int(len(order_data) / n_partitions)
        week = order_data[0][-1] - 1
        uc = UseCasePartition(skus_ini, all_skus, self.layout_path,
                              self.n_levels, current_week=week)
        use_case_partitions = [uc]
        # for order in order_data:
        #     if uc.last_order - uc.first_order >= part_size:
        #         # start a new partition
        #         uc = UseCasePartition(uc.initial_skus[uc.current_week].copy(),
        #                               uc.distinct_skus, self.layout_path,
        #                               self.n_levels, uc.current_week,
        #                               uc.last_order)
        #         use_case_partitions.append(uc)
        #     uc.add_order(order)

        for order in order_data:
            uc.add_order(order)
        return use_case_partitions
    
    def get_initial_partitions_data(self, n_partitions: int):
        import os
        from copy import deepcopy

        root_dir = sep.join([sep.join(
            abspath(__file__).split(sep)[:-1]), "use_cases", self.use_case_name])

        partition_dir = join(root_dir, "partitions")
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)

        skus_ini, _ = self.load_initial_skus(None)
        order_data = self.load_orders(None)
        part_size = int(len(order_data) / n_partitions)

        i = 0
        order_partitions = []
        partition = []
        for order in order_data:
            if i >= part_size:
                order_partitions.append(partition)
                partition = []
                i = 0
            partition.append(order)
            i += 1

        if partition:
            order_partitions.append(partition)

        # ini_fill_lvl = (sum(skus_ini.values()) / 19512)
        initial_pallets_path = f"partitions/0_partition_fill_lvl.json"
        with open(join(root_dir, initial_pallets_path),
                  'w', encoding='utf8') as json_file:
            json.dump(skus_ini, json_file, ensure_ascii=False)
        skus = deepcopy(skus_ini)
        pt = 0
        for orders in order_partitions:
            fill_levels = []
            for order in orders:
                if order[0] == "retrieval":
                    skus[order[1]] -= 1
                elif order[0] == "delivery":
                    skus[order[1]] += 1
                fill_levels.append((sum(skus.values()) / 19512))
            if max(fill_levels) <= 1:
                partition_order_path = f"partitions/{pt}_partition_orders.json"
                partition_pallets_path = f"partitions/{pt+1}_partition_fill_lvl.json"
                with open(join(root_dir, partition_pallets_path),
                          'w', encoding='utf8') as json_file:
                    json.dump(skus, json_file, ensure_ascii=False)
                with open(join(root_dir, partition_order_path),
                          'w', encoding='utf8') as json_file:
                    json.dump(orders, json_file, ensure_ascii=False)
                pt += 1

    @staticmethod
    def get_unique_skus(
            orders: List[Tuple[str, int, int, int, int]]) -> Set[int]:
        """
        Creates a set of the unique SKUs present over all the orders.

        :param orders: The orders that will be fed into the simulation.
        :return: The set of SKUs over all orders.
        """
        skus = set()
        for order in orders:
            skus.add(order[1])
        return skus

    def load_initial_skus(self, use_case_idx: int):
        if use_case_idx != None:
            root_dir = sep.join([sep.join(
                abspath(__file__).split(sep)[:-1]), "use_cases",
                self.use_case_name])
            partition_order_path = f"partitions/{use_case_idx}_partition_fill_lvl.json"
            with open(join(root_dir, partition_order_path)) as json_file:
                initial_fill_json = json.load(json_file)
        else:
            with open(self.initial_sku_path) as json_file:
                initial_fill_json = json.load(json_file)
        skus_ini = defaultdict(int)
        all_skus = set(skus_ini.keys())
        for sku, amount in initial_fill_json.items():
            skus_ini[int(sku)] = amount
            all_skus.add(int(sku))
        return skus_ini, all_skus

    def load_orders(self, use_case_idx: int):
        if use_case_idx != None:
            root_dir = sep.join([sep.join(
                abspath(__file__).split(sep)[:-1]), "use_cases",
                self.use_case_name])
            partition_order_path = f"partitions/{use_case_idx}_partition_orders.json"
            with open(join(root_dir, partition_order_path)) as json_file:
                order_data = json.load(json_file)
        else:
            with open(self.order_path) as json_file:
                order_data = json.load(json_file)
        return order_data

    @classmethod
    def setup_paths(cls, use_case):
        root_dir = sep.join([sep.join(
            abspath(__file__).split(sep)[:-1]), "use_cases", use_case])
        initial_pallets_path = '3_initial_fill_lvl.json'
        orders_path = '2_orders.json'
        layout_path = '1_layout.csv'
        # MINI_LAYOUT_PATH = sep.join(
        #     ['layouts', 'wepastacks_mini_mid_aisles.csv'])
        return (
            join(root_dir, layout_path),
            join(root_dir, orders_path),
            join(root_dir, initial_pallets_path)
        )

    def select_partition(self, partition_idx, period=1):
        if self.use_case_n_partitions > 1:
            if not partition_idx:
                partition_idx = self.use_case_partition_to_use
        use_case = self.partition_use_case(partition_idx=partition_idx, n_partitions=1)[0]
        self.n_rows = use_case.layout.shape[0]
        self.n_columns = use_case.layout.shape[1]
        self.n_skus = len(use_case.distinct_skus)
        self.all_skus = use_case.distinct_skus
        self.n_orders = len(use_case.order_list)
        self.order_list = use_case.order_list
        period = list(use_case.initial_skus.keys())[0]
        # self.initial_pallets_sku_counts = use_case.initial_skus[period]  # self.order_list[0][-1] / p
        self.initial_pallets_sku_counts, _ = self.load_initial_skus(partition_idx)
        self.n_sources = len(
            np.argwhere(use_case.layout == StorageKeys.SOURCE))
        self.n_sinks = len(np.argwhere(use_case.layout == StorageKeys.SINK))
        self.layout = use_case.layout
        self.n_skus_in = use_case.sku_in_counts
        self.n_skus_out = use_case.sku_out_counts
        self.sku_period = use_case.current_week
        self.desired_fill_level = None
        self.shape = use_case.layout.shape + (use_case.n_levels,)


class UseCasePartition:
    """order = (order_type, sku, time, entrance/exit id, production
    batch/shipping id)"""
    layout = None

    def __init__(self, initial_skus: Dict[int, int], all_skus: Set[int],
                 layout_path, n_levels, current_week: int = 0,
                 first_order: int = 0):
        self.distinct_skus = all_skus
        self.order_list: List[Tuple[str, int, int, int, int]] = []
        self.initial_skus = {current_week: initial_skus,
                             current_week + 1: initial_skus.copy()}
        self.sku_in_counts = {current_week: defaultdict(int)}
        self.sku_out_counts = {current_week: defaultdict(int)}
        if UseCasePartition.layout is None:
            UseCasePartition.layout = UseCasePartition.get_layout(layout_path)
        self.n_levels = n_levels
        self.first_order = first_order
        self.last_order = first_order
        self.current_week = current_week

    @staticmethod
    def get_layout(path: str):
        data = genfromtxt(path, delimiter=',')
        data = data.astype(int)
        shape = data.shape
        if data[0, 0] != -1:
            data[0, 0] = -1
            data = np.transpose(data)
        else:
            data = np.delete(data, shape[1] - 1, axis=1)
        return data

    def add_order(self, order: Tuple[str, int, int, int, int]):
        """
        The order entries from left to right correspond to order_type, SKU,
        arrival time, source/sink (depending on the order type), batch and
        period.

        :param order: The iterable containing order information.
        :return: None.
        """
        if order[-1] != self.current_week:
            # start new future counts
            self.current_week = order[-1]
            self.initial_skus[self.current_week + 1] = (
                self.initial_skus[self.current_week].copy())
            self.sku_in_counts[self.current_week] = defaultdict(int)
            self.sku_out_counts[self.current_week] = defaultdict(int)
        self.order_list.append(tuple(order))
        key = order[1]
        if key not in self.distinct_skus:
            self.distinct_skus.add(key)
        if order[0] == 'delivery':
            self.initial_skus[self.current_week + 1][key] += 1
            self.sku_in_counts[self.current_week][key] += 1
        else:
            assert order[0] == 'retrieval'
            self.initial_skus[self.current_week + 1][key] -= 1
            self.sku_out_counts[self.current_week][key] += 1
        self.last_order += 1