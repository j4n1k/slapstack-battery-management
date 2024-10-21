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
                                                 ConstantTimeGreedyPolicy,
                                                 ClosestOpenLocation, BatchFIFO)

from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy)


def get_episode_env(sim_parameters: SimulationParameters,
                    log_frequency: int, nr_zones: int,
                    logfile_name: str,
                    log_dir='./result_data/'):
    seeds = [56513]
    return SlapEnv(
        sim_parameters, seeds,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones,
            logfile_name=logfile_name),
        action_converters=[BatchFIFO(),
                           ClosestOpenLocation(very_greedy=False)])


def _init_run_loop(simulation_parameters,
                   charging_strategy: Union[FixedChargePolicy|RandomChargePolicy],
                   log_dir):
    th = None
    if isinstance(charging_strategy, FixedChargePolicy):
        th = charging_strategy.charging_threshold
    elif isinstance(charging_strategy, RandomChargePolicy):
        th = "random"
    environment: SlapEnv = get_episode_env(
        sim_parameters=simulation_parameters,
        log_frequency=1000, nr_zones=3, log_dir=log_dir,
        logfile_name=f'{charging_strategy.name}_th{th}')
    loop_controls = LoopControl(environment, steps_per_episode=120)
    # state.state_cache.perform_sanity_check()
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                charging_strategy: Union[FixedChargePolicy|RandomChargePolicy],
                print_freq=0, stop_condition=False,
                log_dir='./result_data_charging/', steps_per_episode=None):
    env, loop_controls = _init_run_loop(
        simulation_parameters, charging_strategy, log_dir)
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
        # FixedChargePolicy(40),
        # FixedChargePolicy(50),
        # FixedChargePolicy(60),
        # FixedChargePolicy(70),
        # FixedChargePolicy(80),
        # FixedChargePolicy(90),
        FixedChargePolicy(100),
        # RandomChargePolicy([40, 50, 60, 70, 80], 1)
    ]

    return charging_strategies

params = SimulationParameters(
    use_case="crossstacks_bm",
    use_case_n_partitions=1,
    use_case_partition_to_use=0,
    n_agvs=21,
    generate_orders=False,
    verbose=False,
    resetting=False,
    initial_pallets_storage_strategy=ClosestOpenLocation(),
    pure_lanes=False,
    # https://logisticsinside.eu/speed-of-warehouse-trucks/
    agv_speed=1.2,
    unit_distance=1.1,
    pallet_shift_penalty_factor=90,  # in seconds
    material_handling_time=45,
    compute_feature_trackers=True,
    n_levels=1,
    door_to_door=True,
    # update_partial_paths=False,
    agv_forks=1,
    charging_thresholds=[40, 50, 60, 70, 80]
)

charging_strategies = get_charging_strategies()

if __name__ == '__main__':
    n_strategies = len(charging_strategies)
    for i in range(n_strategies):
        print(f"Running sim with strategy: {charging_strategies[i].name}")
        run_episode(
                simulation_parameters=params,
                charging_strategy=charging_strategies[i],
                print_freq=100000,
                steps_per_episode=120,
                log_dir=
                f'./result_data_charging_crossstacks'
                )
        # parallelize_heterogeneously(
        #     [run_episode] * n_strategies,
        #     list(zip([params] * n_strategies,                    # params
        #              charging_strategies,                           # policy
        #              [0] * n_strategies,                         # print_freq
        #              [False] * n_strategies,                     # warm_start
        #              ['./result_data_charging'] * n_strategies,
        #              )))
