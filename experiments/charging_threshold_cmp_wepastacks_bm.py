import sys
import time
from typing import List, Union
import os
from os.path import join, abspath, sep
from copy import deepcopy

import pandas as pd
import numpy as np

from experiment_commons import ExperimentLogger, LoopControl, get_partitions_path, delete_partitions_data
from slapstack import SlapEnv
from slapstack.helpers import parallelize_heterogeneously
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.storage_policies import (ClassBasedPopularity,
                                                 ClassBasedCycleTime,
                                                 ClosestOpenLocation, BatchFIFO, StoragePolicy,
                                                 ConstantTimeGreedyPolicy)

from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy, LowTHChargePolicy)


def get_episode_env(sim_parameters: SimulationParameters,
                    log_frequency: int, nr_zones: int,
                    logfile_name: str,
                    log_dir: str,
                    action_converters: list,
                    partitions: list):
    if partitions == None:
        partitions = [None]
    seeds = [56513]
    return SlapEnv(
        sim_parameters, seeds, partitions,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones,
            logfile_name=logfile_name),
        action_converters=action_converters)


def _init_run_loop(simulation_parameters,
                   log_dir,
                   action_converters: list,
                   steps_per_episode: int,
                   partitions: list,
                   charging_strategy: ChargingPolicy):
    pt = simulation_parameters.use_case_partition_to_use
    environment: SlapEnv = get_episode_env(
        sim_parameters=simulation_parameters,
        log_frequency=1000, nr_zones=3, log_dir=log_dir,
        logfile_name=f'pt_{pt}_COL_{charging_strategy.name}_th{charging_strategy.charging_threshold}',
        action_converters=action_converters,
        partitions=partitions)
    loop_controls = LoopControl(environment, steps_per_episode=steps_per_episode)
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                charging_check_strategy,
                partitions: list,
                print_freq=0, stop_condition=False,
                log_dir='', steps_per_episode=None,
                action_converters=None):
    env, loop_controls = _init_run_loop(
        simulation_parameters, log_dir, action_converters, steps_per_episode, partitions, charging_check_strategy)
    parametrization_failure = False
    start = time.time()
    while not loop_controls.done:
        decision_mode = env.core_env.decision_mode
        if decision_mode == "charging_check" or decision_mode == "charging":
            prev_event = env.core_env.previous_event
            action = charging_check_strategy.get_action(loop_controls.state,
                                                        agv_id=prev_event.agv.id)
        else:
            if env.done_during_init:
                raise ValueError("Sim ended during init")
            raise ValueError
        output, reward, loop_controls.done, info, _ = env.step(action)
        if print_freq and loop_controls.n_decisions % print_freq == 0:
            ExperimentLogger.print_episode_info(
                charging_check_strategy.name, start, loop_controls.n_decisions,
                loop_controls.state)
            # state.state_cache.perform_sanity_check()
        loop_controls.n_decisions += 1
        if loop_controls.pbar is not None:
            loop_controls.pbar.update(1)
        if loop_controls.done:
            parametrization_failure = True
            env.core_env.logger.write_logs()
    ExperimentLogger.print_episode_info(
        charging_check_strategy.name, start, loop_controls.n_decisions,
        loop_controls.state)
    return parametrization_failure


def get_charging_strategies():
    charging_strategies = []

    charging_strategies += [
        FixedChargePolicy(30),
        FixedChargePolicy(40),
        FixedChargePolicy(50),
        FixedChargePolicy(60),
        FixedChargePolicy(70),
        FixedChargePolicy(80),
        # FixedChargePolicy(90),
        # FixedChargePolicy(100),
        # RandomChargePolicy([40, 50, 60, 70, 80], 1)
    ]

    return charging_strategies


def get_storage_strategies():
    storage_strategies = []
    storage_strategies += [
       ClosestOpenLocation(very_greedy=False),
    ]
    return storage_strategies


storage_policies = get_storage_strategies()
charging_strategies = get_charging_strategies()

if __name__ == '__main__':
    st = [ClosestOpenLocation(very_greedy=False)]
    n_charging_strategies = len(charging_strategies)
    n_storage_strategies = len(storage_policies)
    partitions_path = get_partitions_path("wepastacks_bm")
    delete_partitions_data(partitions_path)
    # parallel charging strat
    n_partitions = 14
    for pt in range(n_partitions):
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=n_partitions,
            use_case_partition_to_use=pt,
            partition_by_week=True,
            n_agvs=40,
            generate_orders=False,
            verbose=False,
            resetting=False,
            initial_pallets_storage_strategy=ConstantTimeGreedyPolicy(),
            pure_lanes=True,
            n_levels=3,
            # https://logisticsinside.eu/speed-of-warehouse-trucks/
            agv_speed=2,
            unit_distance=1.4,
            pallet_shift_penalty_factor=20,  # in seconds
            compute_feature_trackers=True,
            charging_thresholds=[40, 50, 60, 70, 80],
            charge_during_breaks=True
        )
        parallelize_heterogeneously(
            [run_episode] * n_charging_strategies,
            list(zip([params] * n_charging_strategies,                    # params
                     charging_strategies,
                     [[None]] * n_charging_strategies,  # partitions to cycle
                     [0] * n_charging_strategies,                         # print_freq
                     [False] * n_charging_strategies,   # stop condition
                     ['./result_data_charging_wepa/20cs/charge_in_breaks'] * n_charging_strategies,
                     [None] * n_charging_strategies,
                     [[BatchFIFO(),
                      ClosestOpenLocation(very_greedy=False),
                      LowTHChargePolicy(20)]] * n_charging_strategies
                     )))

    # Strategies with charging
    # all_combinations = [
    #     (params, charging_strategies[j], [0, 1, 2, 3], 100, 120, './result_data_charging_pt_wepa')
    #     for i in range(n_storage_strategies)
    #     for j in range(n_charging_strategies)
    # ]
    # parallelize_heterogeneously(
    #     [run_episode] * len(all_combinations),
    #     all_combinations)

    # Old loop
    # parallelize_heterogeneously(
    #     [run_episode] * n_strategies,
    #     list(zip([params] * n_strategies,                    # params
    #              charging_strategies,                           # policy
    #              [0] * n_strategies,                         # print_freq
    #              [False] * n_strategies,                     # warm_start
    #              ['./result_data_charging_wepa'] * n_strategies,
    #              )))
    #

