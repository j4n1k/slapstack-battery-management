import time
from typing import List, Union
import os
from os.path import join, abspath, sep
from copy import deepcopy

import pandas as pd
import numpy as np

from experiment_commons import ExperimentLogger, LoopControl
from slapstack import SlapEnv
from slapstack.helpers import parallelize_heterogeneously
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.storage_policies import (ClassBasedPopularity,
                                                 ClassBasedCycleTime,
                                                 ClosestOpenLocation, BatchFIFO, StoragePolicy)

from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy)

from stable_baselines3 import DQN


def get_episode_env(sim_parameters: SimulationParameters,
                    storage_strategy: StoragePolicy,
                    log_frequency: int, nr_zones: int,
                    logfile_name: str,
                    log_dir='./result_data/'):
    seeds = [56513]
    if isinstance(sim_parameters.initial_pallets_storage_strategy,
                  ClassBasedPopularity):
        sim_parameters.initial_pallets_storage_strategy = ClassBasedPopularity(
            retrieval_orders_only=False,
            future_counts=True,
            init=True,
            n_zones=nr_zones
        )
    return SlapEnv(
        sim_parameters, seeds,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones,
            logfile_name=logfile_name),
        action_converters=[BatchFIFO(),
                           storage_strategy])


def _init_run_loop(simulation_parameters,
                   charging_strategy: Union[FixedChargePolicy|RandomChargePolicy],
                   storage_strategy: StoragePolicy,
                   log_dir):
    th = None
    if isinstance(charging_strategy, FixedChargePolicy):
        th = charging_strategy.charging_threshold
    elif isinstance(charging_strategy, RandomChargePolicy):
        th = "random"
    if hasattr(storage_strategy, 'n_zones'):
        environment: SlapEnv = get_episode_env(
            sim_parameters=simulation_parameters,
            storage_strategy=storage_strategy,
            log_frequency=1000,
            nr_zones=storage_strategy.n_zones, log_dir=log_dir,
            logfile_name=f'{storage_strategy.name}_{charging_strategy.name}_th{th}')
    else:
        environment: SlapEnv = get_episode_env(
            sim_parameters=simulation_parameters,
            storage_strategy=storage_strategy,
            log_frequency=1000, nr_zones=3, log_dir=log_dir,
            logfile_name=f'{storage_strategy.name}_{charging_strategy.name}_th{th}')
    loop_controls = LoopControl(environment, steps_per_episode=120)
    # state.state_cache.perform_sanity_check()
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                charging_strategy: Union[FixedChargePolicy|RandomChargePolicy],
                storage_strategy: StoragePolicy,
                print_freq=0, stop_condition=False,
                log_dir='./result_data_charging/', steps_per_episode=None):
    env, loop_controls = _init_run_loop(
        simulation_parameters, charging_strategy, storage_strategy, log_dir)
    parametrization_failure = False
    start = time.time()
    while not loop_controls.done:
        if env.core_env.decision_mode == "charging":
            prev_event = env.core_env.previous_event
            action = charging_strategy.get_action(loop_controls.state,
                                                  agv_id=prev_event.agv_id)
        else:
            if env.done_during_init:
                raise ValueError("Sim ended during init")
            raise ValueError
        output, reward, loop_controls.done, info = env.step(action)
        if print_freq and loop_controls.n_decisions % print_freq == 0:
            ExperimentLogger.print_episode_info(
                charging_strategy.name, start, loop_controls.n_decisions,
                loop_controls.state)
            # state.state_cache.perform_sanity_check()
        loop_controls.n_decisions += 1
        if loop_controls.pbar is not None:
            loop_controls.pbar.update(1)
        if loop_controls.done:
            parametrization_failure = True
            env.core_env.logger.write_logs()
        # if not loop_controls.done and stop_condition:
        #     loop_controls.done = loop_controls.stop_prematurely()
        #     if loop_controls.done:
        #         parametrization_failure = True
        #         env.core_env.logger.write_logs()
    ExperimentLogger.print_episode_info(
        charging_strategy.name, start, loop_controls.n_decisions,
        loop_controls.state)
    return parametrization_failure


def get_charging_strategies():
    charging_strategies = []

    charging_strategies += [
        FixedChargePolicy(40),
        FixedChargePolicy(50),
        FixedChargePolicy(60),
        FixedChargePolicy(70),
        FixedChargePolicy(80),
        # FixedChargePolicy(90),
        # FixedChargePolicy(100),
        RandomChargePolicy([40, 50, 60, 70, 80], 1)
    ]

    return charging_strategies


def get_storage_strategies(nr_zones: List[int]):
    storage_strategies = []
    for n_zone in nr_zones:
        storage_strategies += [
            ClassBasedCycleTime(
                n_orders=10000, recalculation_steps=1000, n_zones=n_zone),
            ClassBasedPopularity(
                retrieval_orders_only=False, n_zones=n_zone,
                future_counts=True,
                name=f'allOrdersPopularity_future_z{n_zone}'),
            ClassBasedPopularity(
                retrieval_orders_only=True, n_zones=n_zone,
                future_counts=True,
                name=f'retrievalPopularity_future_z{n_zone}'),
            ClassBasedPopularity(
                retrieval_orders_only=False, n_zones=n_zone,
                future_counts=False, n_orders=10000, recalculation_steps=1000,
                name=f'allOrdersPopularity_past_z{n_zone}'),
            ClassBasedPopularity(
                retrieval_orders_only=True, n_zones=n_zone,
                future_counts=False, n_orders=10000, recalculation_steps=1000,
                name=f'retrievalPopularity_past_z{n_zone}')
        ]
    storage_strategies += [
        ClosestOpenLocation(very_greedy=True),
        ClosestOpenLocation(very_greedy=False),
    ]
    return storage_strategies


storage_policies = get_storage_strategies([2, 3, 5])
charging_strategies = get_charging_strategies()

params = SimulationParameters(
                use_case="wepastacks_bm",
                use_case_n_partitions=1,
                use_case_partition_to_use=0,
                n_agvs=40,
                generate_orders=False,
                verbose=False,
                resetting=False,
                initial_pallets_storage_strategy=ClassBasedPopularity(
                    retrieval_orders_only=False,
                    future_counts=True,
                    init=True,
                    # n_zones changes dynamically based on the slap strategy
                    # in get_episode_env
                    n_zones=2
                ),
                pure_lanes=True,
                n_levels=3,
                # https://logisticsinside.eu/speed-of-warehouse-trucks/
                agv_speed=2,
                unit_distance=1.4,
                pallet_shift_penalty_factor=20,  # in seconds
                compute_feature_trackers=True,
                charging_thresholds=[40, 50, 60, 70, 80],
            )

if __name__ == '__main__':
    n_charging_strategies = len(charging_strategies)
    n_storage_strategies = len(storage_policies)
    # for i in range(n_storage_strategies):
    #     for j in range(n_charging_strategies):
    #         print(f"Running sim with storage strategy: {storage_policies[i].name} "
    #               f"and charging strategy: {charging_strategies[j].name}")
    #         run_episode(
    #                 simulation_parameters=params,
    #                 charging_strategy=charging_strategies[j],
    #                 storage_strategy=storage_policies[i],
    #                 print_freq=100,
    #                 steps_per_episode=120,
    #                 log_dir=
    #                 f'./result_data_charging_wepa'
    #                 )
    all_combinations = [
        (params, charging_strategies[j], storage_policies[i], 100, 120, './result_data_charging_wepa')
        for i in range(n_storage_strategies)
        for j in range(n_charging_strategies)
    ]
    parallelize_heterogeneously(
        [run_episode] * len(all_combinations),
        all_combinations)
    # parallelize_heterogeneously(
    #     [run_episode] * n_strategies,
    #     list(zip([params] * n_strategies,                    # params
    #              charging_strategies,                           # policy
    #              [0] * n_strategies,                         # print_freq
    #              [False] * n_strategies,                     # warm_start
    #              ['./result_data_charging_wepa'] * n_strategies,
    #              )))
    #

