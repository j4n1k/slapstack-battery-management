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
                                                  ChargingPolicy, LowTHChargePolicy)
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

class HStepLookahead:
    def __init__(self, env, horizon: int = 1500):
        """
        H-step lookahead policy for charging threshold decisions.

        Args:
            env: SlapStack environment
            horizon: Number of charging decisions to look ahead
        """
        self.env = env
        self.horizon = horizon
        self.thresholds = env.core_env.state.params.charging_thresholds

    def evaluate_sequence(self, env_copy, action) -> float:
        """Evaluate a sequence of threshold choices by simulating them."""
        total_reward = 0
        done = False
        n_charging_decisions = 0

        while not done:
            decision_mode = env_copy.core_env.decision_mode
            if n_charging_decisions > 100:
                break
            if decision_mode == "charging_check":
                if not action:
                    # after the sampled action is executed follow policy
                    prev_event = env_copy.core_env.previous_event
                    action = LowTHChargePolicy(20).get_action(env_copy.core_env.state,
                                                                agv_id=prev_event.agv.id)
                _, reward, done, _, _ = env_copy.step(action)
                total_reward += env_copy.core_env.state.trackers.average_service_time
                n_charging_decisions += 1
                action = None

            # Break if we've made enough charging decisions
            # if n_charging_decisions >= self.horizon:
            #     break

        # Calculate final metrics
        final_service_time = env_copy.core_env.state.trackers.average_service_time

        # Combine metrics into score (negative since we want to minimize)
        score = -final_service_time

        return score

    def get_action_by_rollout(self, env_state) -> int:
        """Get best action by evaluating H-step sequences."""
        env_copy = pickle.loads(env_state)
        valid_actions = env_copy.valid_action_mask()
        best_action = 0
        best_score = -np.inf

        # Generate all possible H-length sequences
        valid_actions = env_copy.valid_action_mask()
        sequence_scores = []

        # Evaluate each sequence
        for action, valid in enumerate(valid_actions):
            # Create environment copy
            env_copy = pickle.loads(env_state)
            score = -np.inf
            # Evaluate this sequence
            if valid:
                score = self.evaluate_sequence(env_copy, action)
                sequence_scores.append((action, score))


            # Update best action if this sequence is better
            if score > best_score:
                best_score = score
                best_action = action  # Only execute first action

        # Print debug info
        chosen_threshold = self.thresholds[best_action]
        print(f"\nChosen action: {chosen_threshold}")
        return best_action


def run_lookahead_episode(env, simulation_parameters):
    """Run an episode using the H-step lookahead policy."""
    policy = HStepLookahead(env, horizon=3)
    done = False
    total_reward = 0
    n_decisions = 0

    while not done:
        reward = 0
        if env.core_env.decision_mode == "charging_check":
            # Pickle current state
            state_pickle = pickle.dumps(env)

            # Get action from lookahead policy
            action = policy.get_action_by_rollout(state_pickle)

            state, reward, done, _, _ = env.step(action)
        # else:
        #     # Default action for non-charging decisions
        #     state, reward, done, _, _ = env.step(0)

        total_reward += reward
        n_decisions += 1

        if n_decisions % 100 == 0:
            print(f"Decisions made: {n_decisions}")
            print(f"Current service time: {env.core_env.state.trackers.average_service_time:.2f}")

    return env.core_env.state.trackers.average_service_time


if __name__ == '__main__':
    # Your existing setup code
    params = SimulationParameters(
        use_case="wepastacks_bm",
        use_case_n_partitions=20,
        use_case_partition_to_use=10,
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
        charging_thresholds=[0, 100],
        partition_by_day=True,
        battery_capacity=10,
        charge_during_breaks=False
    )

    environment = get_env(
        sim_parameters=params,
        log_frequency=1000,
        nr_zones=3,
        log_dir='./result_data_charging_cross',
        logfile_name='lookahead_minislap'
    )

    total_reward = run_lookahead_episode(environment, params)
    print(f"Final reward: {total_reward}")