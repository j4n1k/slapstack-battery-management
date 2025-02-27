import pickle
import time
import shutil
import os
from os.path import exists,  abspath, sep

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, TD3, PPO, SAC
from tqdm import tqdm

from slapstack import SlapEnv

from slapstack.core_state import State, Trackers
from slapstack.core_state_location_manager import LocationManager
from slapstack.helpers import create_folders, TravelEventKeys
from slapstack.interface_templates import SlapLogger, SimulationParameters
from slapstack_controls.charging_policies import ChargingPolicy
from slapstack_controls.storage_policies import (
    StoragePolicy, ClosestOpenLocation, ClosestToNextRetrieval, ShortestLeg,
    BatchFIFO, ClassBasedPopularity, ShortestLeg)


class ExperimentLogger(SlapLogger):
    def __init__(self, filepath: str, logfile_name: str = 'experiment_data',
                 n_steps_between_saves=10000, nr_zones=3):
        super().__init__(filepath)
        self.n_steps_between_saves = n_steps_between_saves
        self.log_dir = filepath
        create_folders(f'{self.log_dir}/dummy')
        self.logfile_name = logfile_name
        self.log_data = []
        self.prev_n_orders = 0
        self.n_zones = nr_zones
        self.t_s = time.time()

    def set_logfile_name(self, logfile: str):
        self.logfile_name = logfile

    def log_state(self):
        s = self.slap_state
        first_step = len(self.log_data) == 0
        save_logs = len(self.log_data) % self.n_steps_between_saves == 0
        n_orders = len(s.trackers.finished_orders)
        if n_orders != self.prev_n_orders:
            self.prev_n_orders = n_orders
            self.log_data.append(ExperimentLogger.__get_row(s, self.t_s))
            if (not first_step and save_logs) or s.done:
                self.write_logs()

    def write_logs(self):
        n_orders = len(self.slap_state.trackers.finished_orders)
        cols = self.__get_header(self.slap_state)
        df = pd.DataFrame(data=self.log_data,
                          columns=cols)
        df.to_csv(f'{self.log_dir}/{self.logfile_name}_{n_orders}.csv')
        self.log_data = []

    def get_log(self):
        cols = self.__get_header(self.slap_state)
        df = pd.DataFrame(data=self.log_data,
                          columns=cols)
        return df

    @staticmethod
    def __get_header(state: State):
        zm = state.location_manager.zone_manager
        header = [
            # Travel Info
            'total_distance',
            'average_distance',
            'travel_time_retrieval_ave',
            'distance_retrieval_ave',
            'total_shift_distance',
            'utilization_time',
            # Order Info
            'n_queued_retrieval_orders',
            'n_queued_delivery_orders',
            'n_finished_orders',
            # KPIs
            'kpi__throughput',
            'kpi__makespan',
            'kpi__average_service_time',
            'kpi__cycle_time',
            # Broad Trackers
            'runtime',
            'n_free_agvs',
            'n_pallet_shifts',
            'n_steps',
            'n_decision_steps',
            'fill_level',
            'entropy',
            # Charging Trackers
            'n_queued_charging_events',
            'avg_battery_level',
            'n_agv_depleted',
            'n_agv_not_depleted',
            'n_charging_events'
        ]
        if len(zm.n_open_locations_per_zone) != 0:
            header += [f'fill_zone_{i}'
                       for i in range(len(zm.n_open_locations_per_zone.keys()))]
        return header

    @staticmethod
    def __get_row(s: State, t_s=0.0):
        zm = s.location_manager.zone_manager
        tes = s.trackers.travel_event_statistics
        t = s.trackers
        sc = s.location_manager
        am = s.agv_manager
        row = (
            # Travel Info:
            tes.total_distance_traveled,
            tes.average_travel_distance(),
            tes.get_average_travel_time_retrieval(),
            tes.get_average_travel_distance_retrieval(),
            tes.total_shift_distance,
            am.get_average_utilization() / s.time if s.time != 0 else 0,
            # Order Info:
            t.n_queued_retrieval_orders,
            t.n_queued_delivery_orders,
            len(t.finished_orders),
            # KPIs
            ExperimentLogger.__get_throughput(s),  # throughput
            s.time,  # makespan
            t.average_service_time,
            ExperimentLogger.__get_cycle_time(sc),
            # Broad Trackers
            time.time() - t_s,
            am.n_free_agvs,
            t.number_of_pallet_shifts,
            s.n_steps + s.n_silent_steps,
            s.n_steps,
            t.get_fill_level(),
            ExperimentLogger.__get_lane_entropy(sc),
            # Charging Trackers
            sum(len(lst) for lst in s.agv_manager.queued_charging_events.values()),
            am.get_average_agv_battery(),
            am.get_n_depleted_agvs(),
            am.get_n_charged_agvs(),
            am.get_n_charging_events()

        )
        fill_level_per_zone = tuple(
                1 - np.array(list(zm.n_open_locations_per_zone.values())) /
                np.array(list(zm.n_total_locations_per_zone.values()))
        )
        return row + fill_level_per_zone

    @staticmethod
    def __get_cycle_time(sc: LocationManager):
        sku_cycle_times = sc.sku_cycle_time
        sum_cycle_times = 0
        if len(sku_cycle_times) != 0:
            for sku, cycle_time in sku_cycle_times.items():
                sum_cycle_times += cycle_time
            return sum_cycle_times / len(sku_cycle_times)
        else:
            return 0

    @staticmethod
    def __get_lane_entropy(sc: LocationManager):
        lane_entropies = sc.lane_wise_entropies
        average_entropy = 0
        for lane, entropy in lane_entropies.items():
            average_entropy += entropy
        return average_entropy / len(lane_entropies)

    @staticmethod
    def __get_throughput(s: State):
        t = s.trackers
        return len(t.finished_orders) / s.time if s.time != 0 else 0

    @staticmethod
    def print_episode_info(strategy_name: str, episode_start_time: float,
                           episode_decisions: int, end_state: State):
        zm = end_state.location_manager.zone_manager
        tes = end_state.trackers.travel_event_statistics
        t = end_state.trackers
        fill_level_per_zone = \
            1 - np.array(list(zm.n_open_locations_per_zone.values())) / \
            np.array(list(zm.n_total_locations_per_zone.values()))

        sc = end_state.location_manager
        es = end_state
        print(f"Episode with storage strategy "
              f"{strategy_name} ended after "
              f"{time.time() - episode_start_time} seconds:")
        print(f"\tBroad Trackers:")
        print(f"\t\tNumber of decisions: {episode_decisions}")
        print(f"\t\tNumber of pallet shifts: {t.number_of_pallet_shifts}")
        print(f"\t\tShift distance penalty: {tes.total_shift_distance}")
        print(f"\t\tFill Level: {t.get_fill_level()}")
        print(f"\t\tFill level per zone: {fill_level_per_zone}")
        print(f'\t\tAverage Lane Entropy: '
              f'{ExperimentLogger.__get_lane_entropy(sc)}')
        print(f'\tKPI:')
        print(f'\t\tThroughput: {ExperimentLogger.__get_throughput(end_state)}')
        print(f"\t\tMakespan: {end_state.time}")
        print(f"\t\tMean service time: {t.average_service_time}")
        print(f'\t\tCycle Time: {ExperimentLogger.__get_cycle_time(sc)}')
        print("\tTravel Info:")
        print(f"\t\tTotal travel distance: {tes.total_distance_traveled}")
        print(f"\t\tMean travel distance: {tes.average_travel_distance()}")
        print(f"\t\tMean travel time: {tes.average_travel_time()}")
        td_rl1 = tes.average_travel_distance(TravelEventKeys.RETRIEVAL_1STLEG)
        td_rl2 = tes.average_travel_distance(TravelEventKeys.RETRIEVAL_2ND_LEG)
        mean_dist_ret = (td_rl1 + td_rl2) / 2
        print(f"\t\tMean travel distance retrieval: {mean_dist_ret}")
        tt_rl1 = tes.average_travel_time(TravelEventKeys.RETRIEVAL_2ND_LEG)
        tt_rl2 = tes.average_travel_time(TravelEventKeys.RETRIEVAL_1STLEG)
        mean_time_ret = (tt_rl1 + tt_rl2) / 2
        print(f"\t\tMean travel time retrieval: {mean_time_ret}")
        print(f'\t\tAverage AGV utilization: '
              f'{end_state.agv_manager.get_average_utilization() / es.time}')
        print(f"\tOrder Info:")
        print(f"\t\tPending Retrieval Orders: {t.n_queued_retrieval_orders}")
        print(f"\t\tPending Delivery Orders: {t.n_queued_delivery_orders}")
        print(f"\t\tNumber of orders completed: {len(t.finished_orders)}")
        print(f"\t\tNumber of Visible AGVs: {es.agv_manager.n_visible_agvs}")


class LoopControl:
    def __init__(self, env: SlapEnv, pbar_on=True, steps_per_episode=None):
        self.done = False
        self.n_decisions = 0
        if pbar_on:
            if not steps_per_episode:
                total_orders = env.core_env.orders.n_orders
                finished_orders = len(env.core_env.state.trackers.finished_orders)
                remaining_orders = total_orders - finished_orders
                self.pbar = tqdm(total=int(remaining_orders / 2))
            else:
                total_orders = env.core_env.orders.n_orders
                finished_orders = total_orders / steps_per_episode
                remaining_orders = total_orders - finished_orders
                self.pbar = tqdm(total=int(steps_per_episode))
        else:
            self.pbar = None
        self.state: State = env.core_env.state
        self.trackers: Trackers = env.core_env.state.trackers

    def stop_prematurely(self):
        t = self.state.trackers
        if (t.n_queued_delivery_orders > 240
                or t.n_queued_retrieval_orders > 330):
            return True
        # if (t.n_queued_delivery_orders > 627
        #         or t.n_queued_retrieval_orders > 693
        #         or t.average_service_time > 1800):
        #     return True
        return False


def _init_run_loop(simulation_parameters, storage_strategy, log_dir):
    week = simulation_parameters.use_case_partition_to_use
    if hasattr(storage_strategy, 'n_zones'):
        environment: SlapEnv = get_episode_env(
            sim_parameters=simulation_parameters,
            log_frequency=1000,
            nr_zones=storage_strategy.n_zones, log_dir=log_dir)
    else:
        environment: SlapEnv = get_episode_env(
            sim_parameters=simulation_parameters,
            log_frequency=1000, nr_zones=3, log_dir=log_dir)
    loop_controls = LoopControl(environment)
    # state.state_cache.perform_sanity_check()
    environment.core_env.logger.set_logfile_name(
        f'pt_{week}_COL_nointerrupt_th100_n{simulation_parameters.n_agvs}_n{simulation_parameters.n_cs}')
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                storage_strategy: StoragePolicy,
                charging_strategy: ChargingPolicy,
                charging_check_strategy: ChargingPolicy,
                print_freq=0,
                warm_start=False, log_dir='./result_data/',
                stop_condition=False, pickle_at_decisions=np.infty,
                testing=False,
                ):
    pickle_path = (f'end_env_{storage_strategy.name}_'
                   f'{pickle_at_decisions}.pickle')
    env, loop_controls = _init_run_loop(
        simulation_parameters, storage_strategy, log_dir)
    parametrization_failure = False
    start = time.time()
    if exists(pickle_path):
        env = pickle.load(open(pickle_path, 'rb'))
        loop_controls = LoopControl(env, pbar_on=True)
    while not loop_controls.done:
        if env.core_env.decision_mode == "charging":
            prev_event = env.core_env.previous_event
            action = charging_strategy.get_action(loop_controls.state,
                                                  agv_id=prev_event.agv_id)
        elif env.core_env.decision_mode == "charging_check":
            prev_event = env.core_env.previous_event
            action = charging_check_strategy.get_action(loop_controls.state,
                                                  agv_id=prev_event.agv.id)
        elif warm_start and len(loop_controls.trackers.finished_orders) < 1000:
            action = ClosestOpenLocation().get_action(loop_controls.state)
        elif (isinstance(storage_strategy, ClosestToNextRetrieval)
              or isinstance(storage_strategy, ShortestLeg)):
            action = storage_strategy.get_action(
                loop_controls.state, env.core_env)
        else:
            action = storage_strategy.get_action(loop_controls.state)
        output, reward, loop_controls.done, info, _ = env.step(action)
        if print_freq and loop_controls.n_decisions % print_freq == 0:
            if loop_controls.n_decisions > pickle_at_decisions:
                pickle.dump(env, open(pickle_path, 'wb'))
            ExperimentLogger.print_episode_info(
                storage_strategy.name, start, loop_controls.n_decisions,
                loop_controls.state)
            # state.state_cache.perform_sanity_check()
        loop_controls.n_decisions += 1
        if loop_controls.pbar is not None:
            loop_controls.pbar.update(1)
        if not loop_controls.done and stop_condition:
            # will set the done control to true is stop criteria is met
            loop_controls.done = loop_controls.stop_prematurely()
            if loop_controls.done:
                parametrization_failure = True
                env.core_env.logger.write_logs()
    ExperimentLogger.print_episode_info(
        storage_strategy.name, start, loop_controls.n_decisions,
        loop_controls.state)
    if not testing:
        return parametrization_failure
    else:
        return loop_controls.state


def get_episode_env(sim_parameters: SimulationParameters,
                    log_frequency: int, nr_zones: int,
                    log_dir='./result_data/',
                    partitions=None):
    seeds = [56513]
    if partitions is None:
        partitions = [None]
    if isinstance(sim_parameters.initial_pallets_storage_strategy,
                  ClassBasedPopularity):
        sim_parameters.initial_pallets_storage_strategy = ClassBasedPopularity(
            retrieval_orders_only=False,
            future_counts=True,
            init=True,
            n_zones=nr_zones
        )
    return SlapEnv(
        sim_parameters, seeds, partitions,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones),
        action_converters=[BatchFIFO()])

def create_output_str(model, net_arch):
    output_string = None
    if isinstance(model, DQN):
        str_net = [str(layer) for layer in net_arch]
        net = "_".join(str_net)
        buffer = model.buffer_size
        batch = model.batch_size
        e = model.exploration_fraction
        tau = model.tau
        update = model.target_update_interval
        output_string = f"DQN_{buffer}_{batch}_{e}_{net}_{update}_{tau}"
    elif isinstance(model, TD3):
        str_net = [str(layer) for layer in net_arch]
        net = "_".join(str_net)
        buffer = model.buffer_size
        batch = model.batch_size
        tau = model.tau
        output_string = f"TD3_{buffer}_{batch}__{net}_{tau}"
    elif isinstance(model, PPO):
        str_net = [str(layer) for layer in net_arch]
        net = "_".join(str_net)
        n_steps = model.n_steps
        ent_coef = model.ent_coef
        batch = model.batch_size
        output_string = f"PPO{n_steps}_{batch}_{net}_{ent_coef}"
    elif isinstance(model, SAC):
        str_net = [str(layer) for layer in net_arch]
        net = "_".join(str_net)
        buffer = model.buffer_size
        batch = model.batch_size
        tau = model.tau
        update = model.target_update_interval
        output_string = f"SAC_{buffer}_{batch}_{net}_{update}_{tau}"
    return output_string


def gen_charging_stations(layout, n_cs) -> pd.DataFrame:
    charging_locs = [len(layout.columns) * i // (n_cs + 1) for i in range(1, n_cs + 1)]
    aisle = pd.DataFrame({i: -1 if i == 0 or i == len(layout.columns) - 1 else -6 if i in charging_locs else -2 for i in
                          range(len(layout.columns))}, index=[1])
    aisle1 = pd.DataFrame(
        {i: -1 if i == 0 or i == len(layout.columns) - 1 else -5 if i in charging_locs else -2 for i in
         range(len(layout.columns))}, index=[2])
    line = pd.DataFrame(
        {i: -1 if i == 0 or i == len(layout.columns) - 1 else -2 if i == 1 else -5 for i in range(len(layout.columns))},
        index=[3])
    aisle2 = pd.DataFrame(
        {i: -1 if i == 0 or i == len(layout.columns) - 1 else -5 if i == 2 else -2 for i in range(len(layout.columns))},
        index=[4])

    layout_new = pd.concat([layout.iloc[:1], aisle, aisle1, line, aisle2, layout.iloc[1:]]).reset_index(drop=True)
    return layout_new


def gen_charging_stations_left(layout, n_cs) -> pd.DataFrame:
    # Calculate the positions for charging stations
    charging_locs = [len(layout.index) * i // (n_cs + 1) for i in range(1, n_cs + 1)]

    # Create new columns for the charging stations and aisles
    aisle = pd.Series({i: -1 if i == 0 or i == len(layout.index) - 1 else -6 if i in charging_locs else -2 for i in
                       range(len(layout.index))})
    aisle1 = pd.Series({i: -1 if i == 0 or i == len(layout.index) - 1 else -5 if i in charging_locs else -2 for i in
                        range(len(layout.index))})
    # line = pd.Series({i: -1 if i == 0 or i == len(layout.index)-1 else -2 if i == 1 else -5 for i in range(len(layout.index))})
    # aisle2 = pd.Series({i: -1 if i == 0 or i == len(layout.index)-1 else -5 if i in charging_locs else -2 for i in range(len(layout.index))})
    # aisle3 = pd.Series({i: -1 if i == 0 or i == len(layout.index)-1 else -5 if i in charging_locs else -2 for i in range(len(layout.index))})

    # Concatenate the new columns with the existing layout, preserving the structure
    layout_new = pd.concat([
        layout.iloc[:, :1],  # First column of original layout
        pd.DataFrame({0: aisle, 1: aisle1}),  # New columns
        layout.iloc[:, 2:]  # Rest of the original layout
    ], axis=1)

    # Reset and rename the columns
    layout_new.columns = range(len(layout_new.columns))

    return layout_new


def get_layout_path(use_case_base="wepastacks"):
    # Get the directory of the current file (trainer.py)
    current_dir = sep.join(abspath(__file__).split(sep)[:-1])

    # Navigate up to the slapstack root directory
    slapstack_dir = sep.join(current_dir.split(sep)[:-1])

    # Construct the path to 1_layout.csv
    layout_path = sep.join([
        slapstack_dir,
        "1_environment",
        "slapstack",
        "slapstack",
        "use_cases",
        use_case_base,
        '1_layout.csv'
    ])

    # Verify that the file exists
    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"Layout file not found at {layout_path}")

    return layout_path

def count_charging_stations(df):
    return (df == -6).sum().sum()

def delete_prec_dima(folder_path, use_case):
    files_to_delete = [f'predecessors_{use_case}.npy', f'distance_matrix_{use_case}.npy']
    for filename in files_to_delete:
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")


def delete_partitions_data(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def get_partitions_path(use_case):
    # Get the directory of the current file (trainer.py)
    current_dir = sep.join(abspath(__file__).split(sep)[:-1])

    # Navigate up to the slapstack root directory
    slapstack_dir = sep.join(current_dir.split(sep)[:-1])

    partitions_path = sep.join([
        slapstack_dir,
        "1_environment",
        "slapstack",
        "slapstack",
        "use_cases",
        use_case,
        'partitions'
    ])
    return partitions_path