import pickle
from copy import deepcopy

import numpy as np
from slapstack import SlapEnv
from slapstack.helpers import faster_deepcopy
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.output_converters import FeatureConverterCharging
from slapstack_controls.storage_policies import ClosestOpenLocation, ConstantTimeGreedyPolicy, BatchFIFO
from slapstack_controls.charging_policies import (FixedChargePolicy,
                                                  RandomChargePolicy,
                                                  FullChargePolicy,
                                                  ChargingPolicy)
from experiment_commons import (ExperimentLogger, LoopControl,
                                create_output_str, count_charging_stations, delete_prec_dima, gen_charging_stations,
                                gen_charging_stations_left, get_layout_path, delete_partitions_data,
                                get_partitions_path)

def get_env(sim_parameters: SimulationParameters,
            log_frequency: int,
            nr_zones: int, logfile_name: str, log_dir: str,
            partitions=None, reward_setting=1, state_converter=True):
    if partitions is None:
        partitions = [None]
    seeds = [56513]
    if state_converter:
        return SlapEnv(
            sim_parameters, seeds, partitions,
            logger=ExperimentLogger(
                filepath=log_dir,
                n_steps_between_saves=log_frequency,
                nr_zones=nr_zones,
                logfile_name=logfile_name),
            state_converter=FeatureConverterCharging(
                ["n_depleted_agvs", "avg_battery", "utilization",
                 "queue_len_charging_station", "global_fill_level",
                 "queue_len_retrieval_orders", "queue_len_delivery_orders"],
                reward_setting=reward_setting),
            action_converters=[BatchFIFO(),
                               ClosestOpenLocation(very_greedy=False),
                               FixedChargePolicy(100)]
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
                               ClosestOpenLocation(very_greedy=False),
                               FixedChargePolicy(100)])


def _init_run_loop(simulation_parameters, name, log_dir, state_converter=True):
    pt = simulation_parameters.use_case_partition_to_use
    environment: SlapEnv = get_env(
        sim_parameters=simulation_parameters,
        log_frequency=1000, nr_zones=3, log_dir=log_dir,
        #logfile_name=f'{name}_th{name}'
        logfile_name=f'', state_converter=state_converter)
    loop_controls = LoopControl(environment, steps_per_episode=160)
    return environment, loop_controls


# def run_episode(simulation_parameters: SimulationParameters,
#                 model,
#                 cfg,
#                 print_freq=0,
#                 log_dir='',
#                 writer=None,
#                 state_converter=True):
#     name = cfg.model.agent.name
#     #df_actions = pd.DataFrame(columns=["Step", "Action", "kpi__makespan"])
#     df_actions = pd.DataFrame()
#     env, loop_controls = _init_run_loop(
#         simulation_parameters, name, log_dir, state_converter)
#     loop_controls.state = env.core_env.state
#     pt_idx = simulation_parameters.use_case_partition_to_use
#     parametrization_failure = False
#     start = time.time()
#     while not loop_controls.done:
#         if env.core_env.decision_mode == "charging":
#             prev_event = env.core_env.previous_event
#             state_repr = env.current_state_repr
#             if isinstance(model, ChargingPolicy):
#                 action = model.get_action(loop_controls.state,
#                                                       agv_id=prev_event.agv_id)
#             else:
#                 action, state = model.predict(state_repr,
#                                               deterministic=True)
#         elif env.core_env.decision_mode == "charging_check":
#             prev_event = env.core_env.previous_event
#             state_repr = env.current_state_repr
#             action = model.predict(state_repr,
#                                    deterministic=True)
#         else:
#             raise ValueError
#         output, reward, loop_controls.done, info = env.step(action)
#         if print_freq and loop_controls.n_decisions % print_freq == 0:
#             ExperimentLogger.print_episode_info(
#                 name, start, loop_controls.n_decisions,
#                 loop_controls.state)
#         loop_controls.n_decisions += 1
#         if isinstance(model, SAC):
#             action = action[0]
#         logs = env.core_env.logger.get_log()
#         am = env.core_env.state.agv_manager
#         s = env.core_env.state
#         action_taken = pd.DataFrame(data={
#             "Step": [loop_controls.n_decisions],
#             "Action": [action[0] if isinstance(model, SAC) else action],
#             "kpi__makespan": [s.time],
#             "kpi__average_service_time": [s.trackers.average_service_time],
#             "avg_battery_level": [am.get_average_agv_battery()],
#             "n_queued_charging_events": [sum(len(lst) for lst in am.queued_charging_events.values())],
#             "n_queued_retrieval_orders": [s.trackers.n_queued_retrieval_orders],
#             "n_depleted_agvs": [am.get_n_depleted_agvs()]
#         })
#         writer.add_scalar(f'Evaluation/{pt_idx}/Makespan', s.time, pt_idx)
#         writer.add_scalar(f'Evaluation/{pt_idx}/Servicetime', s.trackers.average_service_time, pt_idx)
#         writer.add_scalar(f'Evaluation/{pt_idx}/Avg_Battery_Level', am.get_average_agv_battery(), pt_idx)
#         writer.add_scalar(f'Evaluation/{pt_idx}/N_Charging_Events', sum(len(lst) for lst in am.queued_charging_events.values()), pt_idx)
#         writer.add_scalar(f'Evaluation/{pt_idx}/N_Retrieval_Orders', s.trackers.n_queued_retrieval_orders, pt_idx)
#         writer.add_scalar(f'Evaluation/{pt_idx}/N_Depleted_AGV', am.get_n_depleted_agvs(), pt_idx)
#
#         if isinstance(model, SAC):
#             action = action.item()
#         elif isinstance(model, DQN):
#             action = cfg.task.task.charging_thresholds[action.item()]
#         else:
#             action = cfg.model.agent.model_params.threshold
#         writer.add_scalar(f'Evaluation/{pt_idx}/Action', action)
#
#         df_actions = pd.concat([df_actions, action_taken])
#         if loop_controls.pbar is not None:
#             loop_controls.pbar.update(1)
#         if loop_controls.done:
#             parametrization_failure = True
#             env.core_env.logger.write_logs()
#     ExperimentLogger.print_episode_info(
#         name, start, loop_controls.n_decisions,
#         loop_controls.state)
#     df_actions.to_csv(log_dir + f'/pt_{pt_idx}_th{name}_actions.csv')
#     return parametrization_failure, df_actions

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state              # Environment state
        self.parent = parent            # Parent node
        self.children = []              # Child nodes
        self.visits = 0                 # Number of times node has been visited
        self.reward = 0                 # Sum of rewards (for backpropagation)
        self.action = action            # Action taken to reach this node

    def expand(self, env):
        """Expand the node by adding children for all possible actions."""
        possible_actions = range(env.action_space.n)
        for action in possible_actions:
            env_copy = deepcopy(env)  # Clone environment to simulate ahead
            next_state, reward, done, info, _ = env_copy.step(action)
            child_node = MCTSNode(state=next_state, parent=self, action=action)
            self.children.append(child_node)
        return self.children

    def select_child(self):
        """Select the child node with the highest UCT value."""
        C = 1.414  # Exploration constant
        uct_values = [
            (child.reward / (child.visits + 1e-6)) + C * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(uct_values)]

    def backpropagate(self, reward):
        """Backpropagate the reward to update statistics."""
        self.visits += 1
        self.reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def best_action(self):
        """Return the action that leads to the most visited child."""
        return max(self.children, key=lambda child: child.visits).action


class MCTS:
    def __init__(self, env, save_dir: str = "./mcts_states"):
        self.env = env
        self.save_dir = save_dir

    def save_state(self, state_id: int) -> str:
        """Save environment state to file"""
        path = f"{self.save_dir}/state_{state_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.env, f)
        return path

    def load_state(self, path: str):
        """Load environment state from file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def run(self, root_state, num_simulations=1):
        root = MCTSNode(state=root_state)

        for _ in range(num_simulations):
            node = root

            # 1. Selection: Traverse the tree until we reach a leaf node
            while node.children:
                node = node.select_child()

            # 2. Expansion: Expand the leaf node
            if not node.children:
                node.expand(self.env)

            # 3. Simulation: Rollout from the new node to get a reward
            reward = self.rollout(node.state)

            # 4. Backpropagation: Propagate the reward back up the tree
            node.backpropagate(reward)

        # Return the best action from the root
        return root.best_action()

    def rollout(self, state):
        """Simulate a random rollout from the state to estimate reward."""
        # env_copy = deepcopy(self.env)  # Clone the environment
        # env_copy.set_state(state)    # Set the environment to this state
        env_copy = pickle.loads(state.state_pickle)
        total_reward = 0
        done = False
        while not done:
            # Rollout with random actions (you can replace this with a policy)
            action = self.env.action_space.sample()
            state, reward, done, info, _ = env_copy.step(action)
            total_reward += reward
        total_reward += env_copy.core_env.state.trackers.average_service_time
        return total_reward

if __name__ == '__main__':
    log_dir_init = './result_data_charging_cross/init'
    log_dir = './result_data_charging_cross'
    logfile_name = f'mtcs_crossstacks'
    params = SimulationParameters(
        use_case="crossstacks_bm",
        use_case_n_partitions=1,
        use_case_partition_to_use=0,
        partition_by_day=True,
        n_agvs=15,
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
        charging_thresholds=[40, 50, 60, 70, 80],
        charge_during_breaks=False,
        battery_capacity=52
    )
    environment: SlapEnv = get_env(
        sim_parameters=params,
        log_frequency=1000, nr_zones=3, log_dir=log_dir,
        logfile_name=logfile_name)
    mcts = MCTS(environment)
    mcts.run(0)