from os.path import sep, abspath, join

import hydra
from gymnasium.vector import AsyncVectorEnv
from omegaconf import DictConfig, OmegaConf
import os
import time
from typing import Dict

import pandas as pd
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3 import DQN, SAC
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
import wandb
# from wandb.integration.sb3 import WandbCallback
from custom_callbacks import WandbCallback, MaskableEvalCallback

from experiment_commons import (ExperimentLogger, LoopControl,
                                create_output_str, count_charging_stations, delete_prec_dima, gen_charging_stations,
                                gen_charging_stations_left, get_layout_path, delete_partitions_data,
                                get_partitions_path)
from slapstack import SlapEnv
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.output_converters import FeatureConverterCharging
from slapstack_controls.storage_policies import ClosestOpenLocation, ConstantTimeGreedyPolicy, BatchFIFO
from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy)

def get_env(sim_parameters: SimulationParameters,
            log_frequency: int,
            nr_zones: int, logfile_name: str, log_dir: str,
            partitions=None, reward_setting=1, state_converter=True, cfg=None):
    if partitions is None:
        partitions = [None]
    seeds = [56513]
    if state_converter:
        if ((cfg.task.task.name == "go_charging" or cfg.task.task.name == "combined")
                and not cfg.model.agent.name == "Threshold"):
            action_converters = [BatchFIFO(),
                                 ClosestOpenLocation(very_greedy=False),
                                 FixedChargePolicy(100)
                                 ]
            feature_list = ["n_depleted_agvs", "avg_battery", "utilization",
                 "queue_len_charging_station", "global_fill_level", "curr_agv_battery",
                 "dist_to_cs", "queue_len_retrieval_orders", "queue_len_delivery_orders"]
            decision_mode = "charging_check"
        else:
            action_converters = [BatchFIFO(),
                                 ClosestOpenLocation(very_greedy=False)
                                 ]
            feature_list = ["n_depleted_agvs", "avg_battery", "utilization",
                 "queue_len_charging_station", "global_fill_level",
                 "queue_len_retrieval_orders", "queue_len_delivery_orders"]
            decision_mode = cfg.task.task.name
        return SlapEnv(
            sim_parameters, seeds, partitions,
            logger=ExperimentLogger(
                filepath=log_dir,
                n_steps_between_saves=log_frequency,
                nr_zones=nr_zones,
                logfile_name=logfile_name),
            state_converter=FeatureConverterCharging(
                feature_list,
                reward_setting=reward_setting, decision_mode=decision_mode),
            action_converters=action_converters
        )
    else:
        return SlapEnv(
            sim_parameters, seeds, partitions,
            logger=ExperimentLogger(
                filepath=log_dir,
                n_steps_between_saves=log_frequency,
                nr_zones=nr_zones,
                logfile_name=logfile_name),
            action_converters=[BatchFIFO(),
                               ClosestOpenLocation(very_greedy=False)])


def _init_run_loop(simulation_parameters, name, log_dir, cfg, state_converter=True):
    pt = simulation_parameters.use_case_partition_to_use
    environment: SlapEnv = get_env(
        sim_parameters=simulation_parameters,
        log_frequency=1000, nr_zones=3, log_dir=log_dir,
        #logfile_name=f'{name}_th{name}'
        logfile_name=f'', state_converter=state_converter, cfg=cfg)
    loop_controls = LoopControl(environment, steps_per_episode=160)
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                model,
                cfg,
                print_freq=0,
                log_dir='',
                writer=None,
                state_converter=True):
    name = cfg.model.agent.name
    #df_actions = pd.DataFrame(columns=["Step", "Action", "kpi__makespan"])
    df_actions = pd.DataFrame()
    env, loop_controls = _init_run_loop(
        simulation_parameters, name, log_dir, cfg, state_converter)
    loop_controls.state = env.core_env.state
    pt_idx = simulation_parameters.use_case_partition_to_use
    parametrization_failure = False
    start = time.time()
    while not loop_controls.done:
        if env.core_env.decision_mode == "charging":
            prev_event = env.core_env.previous_event
            state_repr = env.current_state_repr
            if isinstance(model, ChargingPolicy):
                action = model.get_action(loop_controls.state,
                                                      agv_id=prev_event.agv_id)
            else:
                action, state = model.predict(state_repr,
                                              deterministic=True)
        elif env.core_env.decision_mode == "charging_check":
            prev_event = env.core_env.previous_event
            state_repr = env.current_state_repr
            if isinstance(model, MaskablePPO):
                action_masks = get_action_masks(env)
                action, _states = model.predict(state_repr,
                                                action_masks=action_masks,
                                                deterministic=True)
            else:
                action = model.predict(state_repr,
                                       deterministic=True)
            # action = action[0].item()
        else:
            raise ValueError
        output, reward, loop_controls.done, info, _ = env.step(action.item())
        if print_freq and loop_controls.n_decisions % print_freq == 0:
            ExperimentLogger.print_episode_info(
                name, start, loop_controls.n_decisions,
                loop_controls.state)
        loop_controls.n_decisions += 1
        if isinstance(model, SAC):
            action = action[0]
        logs = env.core_env.logger.get_log()
        am = env.core_env.state.agv_manager
        s = env.core_env.state
        action_taken = pd.DataFrame(data={
            "Step": [loop_controls.n_decisions],
            "Action": [action[0] if isinstance(model, SAC) else action.item()],
            "kpi__makespan": [s.time],
            "kpi__average_service_time": [s.trackers.average_service_time],
            "avg_battery_level": [am.get_average_agv_battery()],
            "n_queued_charging_events": [sum(len(lst) for lst in am.queued_charging_events.values())],
            "n_queued_retrieval_orders": [s.trackers.n_queued_retrieval_orders],
            "n_depleted_agvs": [am.get_n_depleted_agvs()]
        })
        writer.add_scalar(f'Evaluation/{pt_idx}/Makespan', s.time, pt_idx)
        writer.add_scalar(f'Evaluation/{pt_idx}/Servicetime', s.trackers.average_service_time, pt_idx)
        writer.add_scalar(f'Evaluation/{pt_idx}/Avg_Battery_Level', am.get_average_agv_battery(), pt_idx)
        writer.add_scalar(f'Evaluation/{pt_idx}/N_Charging_Events', sum(len(lst) for lst in am.queued_charging_events.values()), pt_idx)
        writer.add_scalar(f'Evaluation/{pt_idx}/N_Retrieval_Orders', s.trackers.n_queued_retrieval_orders, pt_idx)
        writer.add_scalar(f'Evaluation/{pt_idx}/N_Depleted_AGV', am.get_n_depleted_agvs(), pt_idx)
        action = action.item()
        # if isinstance(model, SAC):
        #     action = action.item()
        # elif isinstance(model, DQN):
        #     action = cfg.task.task.charging_thresholds[action[0].item()]
        # else:
        #     action = cfg.model.agent.model_params.threshold
        writer.add_scalar(f'Evaluation/{pt_idx}/Action', action)

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
    return parametrization_failure, df_actions


def run_evaluation_tensorboard(cfg, model, storage_strategy, state_converter=True):
    writer = SummaryWriter(log_dir=cfg.experiment.log_dir)
    n_partitions = cfg.sim_params.use_case_n_partitions
    e_partitions = cfg.evaluation.eval_partitions
    reward = 0
    for pt_idx in e_partitions:
        params = SimulationParameters(
            use_case=cfg.sim_params.use_case,
            use_case_n_partitions=cfg.sim_params.use_case_n_partitions,
            use_case_partition_to_use=pt_idx,
            partition_by_week=cfg.sim_params.partition_by_week,
            n_agvs=cfg.sim_params.n_agvs,
            generate_orders=cfg.sim_params.generate_orders,
            verbose=cfg.sim_params.verbose,
            resetting=cfg.sim_params.resetting,
            initial_pallets_storage_strategy=storage_strategy,
            pure_lanes=cfg.sim_params.pure_lanes,
            agv_speed=cfg.sim_params.agv_speed,
            unit_distance=cfg.sim_params.unit_distance,
            pallet_shift_penalty_factor=cfg.sim_params.pallet_shift_penalty_factor,
            compute_feature_trackers=cfg.sim_params.compute_feature_trackers,
            n_levels=cfg.sim_params.n_levels,
            charging_thresholds=list(cfg.task.task.charging_thresholds),
            charge_during_breaks=cfg.sim_params.charge_during_breaks
        )

        parametrization_failure, episode_results = run_episode(
            simulation_parameters=params,
            cfg=cfg,
            model=model,
            print_freq=100000,
            log_dir=cfg.experiment.log_dir,
            writer=writer,
            state_converter=state_converter
        )

        # Log scalar values
        # writer.add_scalar('Evaluation/Total_Reward', episode_results['kpi__makespan'].iloc[-1], pt_idx)
        #writer.add_scalar('Evaluation/Episode_Length', episode_results['episode_length'], pt_idx)
        # writer.add_scalar('Evaluation/Avg_Battery_Level', episode_results['avg_battery_level'], pt_idx)
        # writer.add_scalar('Evaluation/N_Charging_Events', episode_results['n_queued_charging_events'], pt_idx)
        # writer.add_scalar('Evaluation/N_Retrieval_Orders', episode_results['n_queued_retrieval_orders'], pt_idx)

        # Log action distribution
        action_counts = np.bincount(episode_results['Action'])
        writer.add_histogram('Evaluation/Action_Distribution', action_counts, pt_idx)
        # reward += episode_results["kpi__average_service_time"]

    writer.close()
    return reward / len(e_partitions)


def mask_fn(env: SlapEnv):
    return env.valid_action_mask()


def make_env(sim_params, log_frequency, nr_zones, logfile_name, log_dir, partitions, reward_setting, cfg):
    def _init():
        return get_env(
            sim_parameters=sim_params,
            log_frequency=log_frequency,
            nr_zones=nr_zones,
            log_dir=log_dir,
            logfile_name=logfile_name,
            partitions=partitions,
            reward_setting=reward_setting,
            cfg=cfg
        )
    return _init


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    use_case_base = cfg.sim_params.use_case.split("_")[0]
    layout_path_base = get_layout_path(use_case_base)
    layout_path_present = get_layout_path(cfg.sim_params.use_case)
    partitions_path = get_partitions_path(cfg.sim_params.use_case)
    if cfg.sim_params.use_case == "wepastacks_bm":
        delete_partitions_data(partitions_path)

    # When saving/loading the model:
    best_model_path = os.path.join(BASE_DIR, cfg.experiment.log_dir, cfg.experiment.id, f"{cfg.model.agent.name}/")
    print(OmegaConf.to_yaml(cfg))
    layout_present = pd.read_csv(layout_path_present, header=None, delimiter=",")

    n_cs_present = count_charging_stations(layout_present)
    if n_cs_present != cfg.experiment.n_cs:
        delete_prec_dima(BASE_DIR, cfg.sim_params.use_case)
        use_case_base = cfg.sim_params.use_case.split("_")[0]
        layout_base = pd.read_csv(layout_path_base, header=None, delimiter=",")
        layout_new = pd.DataFrame()
        if cfg.sim_params.use_case == "wepastacks_bm":
            layout_new = gen_charging_stations(layout_base, cfg.experiment.n_cs)
        elif cfg.sim_params.use_case == "crossstacks_bm":
            layout_new = gen_charging_stations_left(layout_base, cfg.experiment.n_cs)
        layout_new.to_csv(layout_path_present,
            header=None, index=False)

    # Wandb Init
    run = wandb.init(
        # name=cfg.experiment.id,
        project="rl-battery-management",
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,
        save_code=False
    )

    # Set up simulation parameters
    storage_strategy = ClosestOpenLocation() if cfg.sim_params.initial_pallets_storage_strategy == "ClosestOpenLocation" else ConstantTimeGreedyPolicy()
    th = list(cfg.task.task.charging_thresholds)
    sim_params = SimulationParameters(
        use_case=cfg.sim_params.use_case,
        use_case_n_partitions=cfg.sim_params.use_case_n_partitions,
        use_case_partition_to_use=cfg.sim_params.use_case_partition_to_use,
        partition_by_week=cfg.sim_params.partition_by_week,
        n_agvs=cfg.sim_params.n_agvs,
        generate_orders=cfg.sim_params.generate_orders,
        verbose=cfg.sim_params.verbose,
        resetting=cfg.sim_params.resetting,
        initial_pallets_storage_strategy=storage_strategy,
        pure_lanes=cfg.sim_params.pure_lanes,
        agv_speed=cfg.sim_params.agv_speed,
        unit_distance=cfg.sim_params.unit_distance,
        pallet_shift_penalty_factor=cfg.sim_params.pallet_shift_penalty_factor,
        compute_feature_trackers=cfg.sim_params.compute_feature_trackers,
        n_levels=cfg.sim_params.n_levels,
        charging_thresholds=th,
        charge_during_breaks=cfg.sim_params.charge_during_breaks
    )

    # Create environment
    env: SlapEnv = get_env(
        sim_parameters=sim_params,
        log_frequency=1000,
        nr_zones=3,
        log_dir=cfg.experiment.log_dir,
        logfile_name=f"{cfg.model.agent.name}_{cfg.experiment.id}",
        reward_setting=cfg.task.task.reward_setting,
        partitions=list(cfg.experiment.t_pt),
        cfg=cfg
    )

    eval_env: SlapEnv = get_env(
        sim_parameters=sim_params,
        log_frequency=1000,
        nr_zones=3,
        log_dir=cfg.experiment.log_dir,
        logfile_name=f"{cfg.model.agent.name}_{cfg.experiment.id}",
        reward_setting=1,
        partitions=[cfg.experiment.e_pt],
        cfg=cfg
    )

    # Create agent
    if cfg.model.agent.name == "PPO":
        env = ActionMasker(env, mask_fn)
        eval_env = ActionMasker(eval_env, mask_fn)
        model = MaskablePPO(MaskableActorCriticPolicy, env,
                            verbose=1,
                            tensorboard_log="./dqn_charging_tensorboard/",
                            ent_coef=cfg.model.agent.model_params.ent_coef,
                            device="cpu")
    elif cfg.model.agent.name == "Threshold":
        model = FixedChargePolicy(charging_threshold=cfg.model.agent.model_params.threshold)
    else:
        raise ValueError(f"Unknown model: {cfg.model.agent.name}")

    # Set up callbacks
    if isinstance(env, ActionMasker):
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=os.path.join(cfg.experiment.log_dir, "eval"),
            eval_freq=cfg.experiment.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=cfg.experiment.n_eval_episodes
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=os.path.join(cfg.experiment.log_dir, "eval"),
            eval_freq=cfg.experiment.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=cfg.experiment.n_eval_episodes
        )


    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=os.path.join(cfg.experiment.log_dir, "checkpoints"),
        name_prefix="rl_model"
    )
    wandb_callback = WandbCallback(
        # model_save_path=f"models/{run.id}",
        verbose=2,
        # log="all",  # Log all variables
        log_interval=1
    )
    # Train the model
    if cfg.model.agent.name != "Threshold":
        if cfg.experiment.setting == "train":
            model.learn(
                total_timesteps=cfg.experiment.total_timesteps,
                callback=[checkpoint_callback, wandb_callback, eval_callback],
                progress_bar=False,
                log_interval=1,
                tb_log_name=f"{cfg.experiment.name}_{cfg.model.agent.name}_{cfg.experiment.id}"
            )
            model.save(f"{cfg.experiment.log_dir}/final_model_{cfg.model.agent.name}_{cfg.experiment.id}.zip")

        # Run evaluation episode
        best_model = model.load(os.path.join(best_model_path, "best_model"))

        mean_eval_reward = run_evaluation_tensorboard(cfg, best_model, storage_strategy)
    else:
        mean_eval_reward = run_evaluation_tensorboard(cfg, model, storage_strategy, state_converter=False)
    wandb.finish()
    print(mean_eval_reward)
    return mean_eval_reward


if __name__ == "__main__":
    main()