import os
import time
from copy import deepcopy
from typing import List, Union, Dict

import gym
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, \
    CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from experiment_commons import (ExperimentLogger, LoopControl, create_output_str)
from slapstack import SlapEnv
from slapstack.helpers import parallelize_heterogeneously
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.output_converters import FeatureConverterCharging
from slapstack_controls.storage_policies import (ClassBasedPopularity,
                                                 ClassBasedCycleTime,
                                                 ClosestOpenLocation, BatchFIFO, ConstantTimeGreedyPolicy)
from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy)
from custom_callbacks import TensorBoardCallback
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from gym.wrappers import FrameStack

def get_eval_env(sim_parameters: SimulationParameters,
                    log_frequency: int, nr_zones: int, logfile_name: str,
                    log_dir='./result_data/'):
    seeds = [56513]
    partitions = [13]
    return SlapEnv(
        sim_parameters, seeds, partitions,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones,
            logfile_name=logfile_name), state_converter=FeatureConverterCharging(
            ["n_depleted_agvs", "avg_battery", "curr_agv_battery", "utilization",
             "queue_len_charging_station",  "global_fill_level",
             "queue_len_retrieval_orders", "queue_len_delivery_orders"], reward_setting=2),
        action_converters=[BatchFIFO(),
                           ClosestOpenLocation(very_greedy=False),
                           ])

def get_episode_env(sim_parameters: SimulationParameters,
                    log_frequency: int, nr_zones: int, logfile_name: str,
                    log_dir='./result_data/', partitions=None):
    if partitions is None:
        partitions = [13]
    seeds = [56513]
    return SlapEnv(
        sim_parameters, seeds, partitions,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones,
            logfile_name=logfile_name), state_converter=FeatureConverterCharging(
            ["n_depleted_agvs", "avg_battery", "curr_agv_battery", "utilization",
             "queue_len_charging_station", "global_fill_level",
             "queue_len_retrieval_orders", "queue_len_delivery_orders"], reward_setting=2),
        action_converters=[BatchFIFO(),
                           ClosestOpenLocation(very_greedy=False),
                           ])
# "time_next_event", "n_depleted_agvs",
#  "avg_battery", "utilization", "avg_entropy",
#  "global_fill_level", "queue_len_charging_station",
#  "queue_len_retrieval_orders", "queue_len_delivery_orders",
#  "throughput_delivery_orders", "throughput_retrieval_orders"
def _init_run_loop(simulation_parameters, name, log_dir):
    pt = simulation_parameters.use_case_partition_to_use
    environment: SlapEnv = get_episode_env(
        sim_parameters=simulation_parameters,
        log_frequency=1000, nr_zones=3, log_dir=log_dir,
        #logfile_name=f'{name}_th{name}'
        logfile_name=f'pt_{pt}_{name}_RL_thSAC')

    loop_controls = LoopControl(environment, steps_per_episode=160)
    # state.state_cache.perform_sanity_check()
    # environment.core_env.logger.set_logfile_name(
    #     f'{name}_n{simulation_parameters.n_agvs}')
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                charging_strategy: DQN,
                print_freq=0,
                log_dir='./result_data_charging_dqn/n_agvs__40/n_cs__4/dqn'):
    name = "SAC"
    df_actions = pd.DataFrame(columns=["Step", "Action", "kpi__makespan"])
    env, loop_controls = _init_run_loop(
        simulation_parameters, name, log_dir)
    loop_controls.state = env.core_env.state
    pt_idx = simulation_parameters.use_case_partition_to_use
    parametrization_failure = False
    start = time.time()
    while not loop_controls.done:
        if env.core_env.decision_mode == "charging":
            prev_event = env.core_env.previous_event
            state_repr = env.current_state_repr
            action, state = charging_strategy.predict(state_repr,
                                                      deterministic=True)
        elif env.core_env.decision_mode == "charging_check":
            prev_event = env.core_env.previous_event
            state_repr = env.current_state_repr
            action = charging_strategy.predict(state_repr,
                                                      deterministic=True)
        else:
            raise ValueError
        output, reward, loop_controls.done, info = env.step(action)
        if print_freq and loop_controls.n_decisions % print_freq == 0:
            ExperimentLogger.print_episode_info(
                name, start, loop_controls.n_decisions,
                loop_controls.state)
            # state.state_cache.perform_sanity_check()
        loop_controls.n_decisions += 1
        if isinstance(charging_strategy, SAC):
            action = action[0]
        action_taken = pd.DataFrame(data={
            "Step": [loop_controls.n_decisions],
            "Action": [action],
            "kpi__makespan": [env.core_env.state.time]})
        df_actions = pd.concat([df_actions, action_taken])
        if loop_controls.pbar is not None:
            loop_controls.pbar.update(1)
        if loop_controls.done:
            parametrization_failure = True
            env.core_env.logger.write_logs()
    ExperimentLogger.print_episode_info(
        name, start, loop_controls.n_decisions,
        loop_controls.state)
    df_actions.to_csv(log_dir + f'/pt_{pt_idx}_th{name}_actions.csv')
    return parametrization_failure


if __name__ == '__main__':
    log_dir_init = './result_data_charging_wepa/init'
    log_dir = './result_data_charging_wepa'
    logfile_name = f'SAC_wepastacks'

    params = SimulationParameters(
        use_case="wepastacks_bm",
        use_case_n_partitions=20,
        use_case_partition_to_use=0,
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
        charging_thresholds=(40, 80),
    )

    env: SlapEnv = get_episode_env(
        sim_parameters=params,
        log_frequency=1000, nr_zones=3, log_dir=log_dir_init,
        logfile_name=logfile_name)

    net_arch = [125, 125]
    policy_kwargs = dict(net_arch=net_arch)
    model = SAC("MlpPolicy",
                env,
                buffer_size=1000,  # 500
                learning_starts=140,
                batch_size=128,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                target_update_interval=1,
                stats_window_size=100,
                tensorboard_log="./dqn_charging_tensorboard/",
                policy_kwargs=policy_kwargs,
                verbose=2,
                seed=None,
                device='auto',
                _init_setup_model=True
                )
    start = time.time()
    output_string = create_output_str(model, net_arch)
    # eval_env = deepcopy(env)
    eval_env = get_eval_env(sim_parameters=params,
        log_frequency=1000, nr_zones=3, log_dir=log_dir_init,
        logfile_name=logfile_name)
    
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./logs/best_model/sac/",
                                 log_path="./logs/best_model/sac/",
                                 eval_freq=1000,
                                 deterministic=True, render=False,
                                 n_eval_episodes=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs/checkpoint/sac/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=20000, progress_bar=False, log_interval=1,
                callback=[eval_callback, checkpoint_callback],
                tb_log_name="R2_charging_wepa" + output_string)
    model.save(f"./model_r2_charging_wepa" + output_string + ".zip")

    # model = model.load("./model_r1_charging_wepaSAC_1000_128_256_256_1_0.005.zip")
    # n_partitions = 20
    # for pt_idx in range(n_partitions):
    #     params = SimulationParameters(
    #         use_case="wepastacks_bm",
    #         use_case_n_partitions=n_partitions,
    #         use_case_partition_to_use=pt_idx,
    #         n_agvs=40,
    #         generate_orders=False,
    #         verbose=False,
    #         resetting=False,
    #         initial_pallets_storage_strategy=ConstantTimeGreedyPolicy(),
    #         pure_lanes=True,
    #         n_levels=3,
    #         # https://logisticsinside.eu/speed-of-warehouse-trucks/
    #         agv_speed=2,
    #         unit_distance=1.4,
    #         pallet_shift_penalty_factor=20,  # in seconds
    #         compute_feature_trackers=True,
    #         charging_thresholds=(40, 80),
    #     )
    #     run_episode(simulation_parameters=params, charging_strategy=model,
    #                 print_freq=100000,
    #                 log_dir=f'./result_data_charging/val_sac')

