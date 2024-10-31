import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
from typing import List, Optional
import wandb
from datetime import datetime

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from experiment_commons import ExperimentLogger
from slapstack import SlapEnv
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.output_converters import FeatureConverterCharging
from slapstack_controls.storage_policies import (
    ClosestOpenLocation, ConstantTimeGreedyPolicy, BatchFIFO
)
from slapstack_controls.charging_policies import FixedChargePolicy


class WandBCallback(BaseCallback):
    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            self.logger.record("train/n_steps", self.num_timesteps)
            wandb.log({
                "train/n_steps": self.num_timesteps,
                "train/reward": self.locals["rewards"][0],
#                "train/episode_length": self.locals["episode_lengths"][0]
            })
        return True


def get_rl_env(sim_parameters: SimulationParameters,
               log_frequency: int,
               nr_zones: int,
               logfile_name: str,
               log_dir: str,
               partitions=None,
               reward_setting=1,
               seed=None):
    if partitions is None:
        partitions = [None]
    if seed is None:
        seed = np.random.randint(0, 100000)
    seeds = [seed]

    action_converters = [BatchFIFO(),
                         ClosestOpenLocation(very_greedy=False),
                         FixedChargePolicy(100)]

    feature_list = ["n_depleted_agvs", "avg_battery", "utilization",
                    "queue_len_charging_station", "global_fill_level",
                    "curr_agv_battery", "dist_to_cs",
                    "queue_len_retrieval_orders", "queue_len_delivery_orders"]

    decision_mode = "charging_check"

    return SlapEnv(
        sim_parameters,
        seeds,
        partitions,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones,
            logfile_name=f"{logfile_name}_{seed}"),
        state_converter=FeatureConverterCharging(
            feature_list,
            reward_setting=reward_setting,
            decision_mode=decision_mode),
        action_converters=action_converters
    )

def make_env(sim_params, log_frequency, nr_zones, logfile_name,
             log_dir, partitions, reward_setting, seed=None):
    def _init():
        env = get_rl_env(
            sim_parameters=sim_params,
            log_frequency=log_frequency,
            nr_zones=nr_zones,
            log_dir=log_dir,
            logfile_name=logfile_name,
            partitions=partitions,
            reward_setting=reward_setting,
            seed=seed
        )
        # Wrap with ActionMasker
        return env

    return _init


@hydra.main(config_path="config", config_name="test_config")
def main(cfg: DictConfig) -> None:
    # Initialize WandB
    run = wandb.init(
        project=cfg.wandb.project_name,
        name=f"subproc_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
    )

    # Set up base parameters
    params = SimulationParameters(
        use_case=cfg.env.use_case,
        use_case_n_partitions=cfg.env.n_partitions,
        use_case_partition_to_use=0,
        n_agvs=cfg.env.n_agvs,
        generate_orders=False,
        verbose=False,
        resetting=False,
        initial_pallets_storage_strategy=ConstantTimeGreedyPolicy(),
        pure_lanes=True,
        n_levels=3,
        agv_speed=2,
        unit_distance=1.4,
        pallet_shift_penalty_factor=20,
        compute_feature_trackers=True,
        charging_thresholds=list(cfg.env.charging_thresholds),
        battery_capacity=cfg.env.battery_capacity,
        partition_by_week=True
    )

    # Feature list for state representation
    # feature_list = [
    #     "n_depleted_agvs", "avg_battery", "utilization",
    #     "queue_len_charging_station", "global_fill_level",
    #     "curr_agv_battery", "dist_to_cs",
    #     "queue_len_retrieval_orders", "queue_len_delivery_orders"
    # ]

    # Create vectorized environment
    vec_env = SubprocVecEnv([
        make_env(
            sim_params=params,
            log_frequency=1000,
            nr_zones=3,
            logfile_name='PPO_test',
            log_dir=cfg.logging.log_dir,
            partitions=[pt],
            reward_setting=1,
        ) for pt in cfg.env.train_partitions
    ])
    vec_env = VecMonitor(vec_env)

    eval_env = SubprocVecEnv([
        make_env(
            sim_params=params,
            log_frequency=1000,
            nr_zones=3,
            logfile_name='PPO_eval',
            log_dir=cfg.logging.eval_log_dir,
            partitions=[pt],
            reward_setting=1,
        ) for pt in cfg.env.eval_partitions
    ])
    eval_env = VecMonitor(eval_env)

    # Initialize PPO model
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        verbose=1,
        tensorboard_log=cfg.logging.tensorboard_dir,
        device=cfg.training.device,
        learning_rate=cfg.training.learning_rate,
        n_steps=cfg.training.n_steps,
        batch_size=cfg.training.batch_size,
        n_epochs=cfg.training.n_epochs,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        clip_range=cfg.training.clip_range,
        ent_coef=cfg.training.ent_coef
    )

    # Callbacks
    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        eval_freq=cfg.evaluation.eval_freq,
        n_eval_episodes=cfg.evaluation.n_eval_episodes,
        best_model_save_path=cfg.logging.best_model_dir,
        log_path=cfg.logging.eval_log_dir,
        deterministic=True,
        verbose=1
    )

    wandb_callback = WandBCallback(log_interval=cfg.logging.log_interval)
    # Training
    try:
        model.learn(
            total_timesteps=cfg.training.total_timesteps,
            progress_bar=False,
            log_interval=cfg.logging.log_interval,
            callback=[wandb_callback, eval_callback],
            tb_log_name=cfg.logging.run_name
        )

        # Save the trained model
        model.save(Path(cfg.logging.model_dir) / f"{cfg.logging.run_name}.zip")

    except Exception as e:
        wandb.alert(
            title="Training Failed",
            text=f"Training failed with error: {str(e)}"
        )
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
