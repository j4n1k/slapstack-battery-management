import os
from typing import Union, Optional

import numpy as np
from stable_baselines3 import DQN, PPO, TD3, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, SubprocVecEnv, VecMonitor
from sb3_contrib.common.maskable.evaluation import evaluate_policy

import gymnasium as gym

import wandb

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self):
        super()._on_step()
        try:
            training_env = self.model.get_env().envs[0].env.unwrapped.gym_env
        except AttributeError:
            training_env = self.model.get_env().envs[0].env.unwrapped
        action = training_env.last_action_taken

        log_dict = {}
        obs = training_env.current_state_repr
        observation = obs.reshape((-1,) + self.model.observation_space.shape)
        observation = obs_as_tensor(observation, self.model.device)
        feature_list = training_env.feature_list
        log_dict.update({f"observations/{f}": observation[0][i].item() for i, f in enumerate(feature_list)})
        # if observation[0][5].item() == 0:
        #     print()
        if isinstance(self.model, DQN):
            with th.no_grad():
                q_values = self.model.q_net(observation)
            q_max = q_values.max().item()
            q_values = q_values.tolist()[0]
            for t, v in enumerate(q_values):
                log_dict[f"train/q_value_{t}"] = v
            log_dict["train/max_q_value"] = q_max

        if isinstance(self.model, PPO) or isinstance(self.model, TD3):
            action = self.__denormalize_action(action)

        agvm = training_env.core_env.state.agv_manager
        queue_per_station = {cs: 0 for cs in agvm.free_charging_stations}
        for cs in agvm.booked_charging_stations.keys():
            queue_per_station[cs] = len(agvm.booked_charging_stations[cs])

        log_dict.update({
            "train/last_charging_action": action,
            "train/average_service_time": training_env.core_env.state.trackers.average_service_time,
            "train/n_retrieval_orders": training_env.core_env.state.trackers.n_queued_retrieval_orders,
            "train/n_delivery_orders": training_env.core_env.state.trackers.n_queued_delivery_orders,
            "train/n_depleted_amr": training_env.core_env.state.agv_manager.get_n_depleted_agvs(),
            "train/utilization": training_env.core_env.state.agv_manager.get_average_utilization() /
                                 training_env.core_env.state.time if training_env.core_env.state.time != 0 else 0,
            "train/queued_charging_events": np.average(list(queue_per_station.values())),
            "train/last_reward": training_env.last_reward,
        })

        wandb.log(log_dict, step=self.num_timesteps)

        if isinstance(self.model, DQN) or isinstance(self.model, SAC):
            if self.model.num_timesteps % self.log_interval == 0:
                self._dump_logs_to_wandb()

        return True

    def _dump_logs_to_wandb(self):
        for key, value in self.model.logger.name_to_value.items():
            wandb.log({f"train/{key}": value}, step=self.num_timesteps)

    def __denormalize_action(self, action):
        # Implement your denormalization logic here
        return action

class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose, log_interval):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self):
        super()._on_step()
        training_env = self.model.get_env().envs[0].env.unwrapped.gym_env
        # prediction, _ = self.model.predict(training_env.current_state_repr)
        action = training_env.last_action_taken
        if isinstance(self.model, DQN):
            obs = training_env.current_state_repr
            observation = obs.reshape((-1,) + self.model.observation_space.shape)
            observation = obs_as_tensor(observation, self.model.device)
            with th.no_grad():
                q_values = self.model.q_net(observation)
            q_max = q_values.max().item()
            self.logger.record("Max Q Value", q_max)

        if isinstance(self.model, PPO) or isinstance(self.model, TD3):
            action = self.__denormalize_action(action)

        # if isinstance(self.model, SAC):
        #     action = action[0]

        agvm = training_env.core_env.state.agv_manager
        queue_per_station = {cs: 0 for cs in agvm.free_charging_stations}
        for cs in agvm.booked_charging_stations.keys():
            queue_per_station[cs] = len(agvm.booked_charging_stations[cs])

        self.logger.record("Last Charging Action",
                           action)
        self.logger.record("Average Service time",
                           training_env.core_env.state.trackers.average_service_time)
        self.logger.record("N Retrieval Orders",
                           training_env.core_env.state.trackers.n_queued_retrieval_orders)
        self.logger.record("N Delivery Orders",
                           training_env.core_env.state.trackers.n_queued_delivery_orders)
        self.logger.record("N Depleted AMR",
                           training_env.core_env.state.agv_manager.get_n_depleted_agvs())
        # self.logger.record("T next event",
        #                    training_env.core_env.state.peek_time_feature)
        self.logger.record("Queued Charging Events",
                           np.average(list(queue_per_station.values())))
        self.logger.record("Last reward", training_env.last_reward)

        if isinstance(self.model, DQN) or isinstance(self.model, SAC):
            if self.model.num_timesteps % self.log_interval == 0:
                self.model._dump_logs()
        return True

    @staticmethod
    def __denormalize_action(y, original_range=(40, 80),
                             target_range=(-1, 1)):
        c, d = target_range
        a, b = original_range
        return ((y - c) * (b - a) / (d - c)) + a


class MaskableEvalCallback(EvalCallback):
    """
    Custom EvalCallback that supports action masking.
    Inherits from the standard EvalCallback.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[EvalCallback] = None,
        callback_after_eval: Optional[EvalCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class ParallelMaskableEvalCallback(MaskableEvalCallback):
    """Custom maskable evaluation callback that supports parallel evaluation environments."""

    def __init__(
            self,
            eval_env: Union[SubprocVecEnv, VecMonitor],
            eval_freq: int = 1000,
            n_eval_episodes: int = 5,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
            deterministic: bool = True,
            verbose: int = 1,
            prefix: str = "eval"
    ):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            deterministic=deterministic,
            verbose=verbose
        )
        self.prefix = prefix

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training statistics across processes
            sync_training_stats = {
                "timesteps": self.num_timesteps,
                "episodes": self._episode_num
            }

            # Conduct maskable evaluations across all parallel environments
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )

            # Calculate mean reward and length across all environments
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)

            # Log to WandB
            wandb.log({
                f"{self.prefix}/mean_reward": mean_reward,
                f"{self.prefix}/std_reward": std_reward,
                f"{self.prefix}/mean_ep_length": mean_length,
                f"{self.prefix}/timesteps": self.num_timesteps
            })

            # Save best model
            if self.best_model_save_path is not None:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.best_model_save_path}")

            # Log to tensorboard and stdout
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_length:.2f}")

        return True