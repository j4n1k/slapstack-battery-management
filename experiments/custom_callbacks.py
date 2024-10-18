import numpy as np
from stable_baselines3 import DQN, PPO, TD3, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
import wandb

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self):
        super()._on_step()
        training_env = self.model.get_env().envs[0].env.unwrapped.gym_env
        action = training_env.last_action_taken

        log_dict = {}
        obs = training_env.current_state_repr
        observation = obs.reshape((-1,) + self.model.observation_space.shape)
        observation = obs_as_tensor(observation, self.model.device)
        feature_list = training_env.feature_list
        log_dict.update({f"observations/{f}": observation[0][i].item() for i, f in enumerate(feature_list)})
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
