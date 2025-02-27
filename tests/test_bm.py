import json
import os
import time
from collections import defaultdict

import pandas as pd
from math import floor
from os.path import join
from unittest import TestCase

import torch
from gymnasium.vector import AsyncVectorEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from torch.utils.data import TensorDataset, DataLoader

from experiments.experiment_commons import run_episode, get_episode_env, get_partitions_path, delete_partitions_data, \
    ExperimentLogger, LoopControl, get_layout_path
from slapstack import SlapCore, SlapEnv
import random
import numpy as np

from slapstack.core_state import State
from slapstack.helpers import print_3d_np
from slapstack.interface_input import Input
from slapstack.interface_templates import SimulationParameters, ChargingStrategy
from slapstack_controls.charging_policies import FixedChargePolicy, LowTHChargePolicy, CombinedChargingPolicy, \
    ChargingPolicy
from slapstack_controls.output_converters import FeatureConverterCharging
from slapstack_controls.storage_policies import ConstantTimeGreedyPolicy, ClosestOpenLocation, BatchFIFO
from tests.experiment_commons import count_charging_stations, delete_prec_dima, gen_charging_stations
from tests.model import train_model


class TestSlapEnv(TestCase):
    @staticmethod
    def get_episode_env(sim_parameters: SimulationParameters,
                        log_frequency: int, nr_zones: int,
                        logfile_name: str,
                        log_dir: str,
                        action_converters: list):
        seeds = [56513]
        return SlapEnv(
            sim_parameters, seeds,
            logger=ExperimentLogger(
                filepath=log_dir,
                n_steps_between_saves=log_frequency,
                nr_zones=nr_zones,
                logfile_name=logfile_name),
            action_converters=action_converters)

    def _init_run_loop(self,
                       simulation_parameters,
                       log_dir,
                       action_converters: list,
                       steps_per_episode: int):
        environment: SlapEnv = self.get_episode_env(
            sim_parameters=simulation_parameters,
            log_frequency=1000, nr_zones=3, log_dir=log_dir,
            logfile_name="go_charging_test",
            action_converters=action_converters)
        loop_controls = LoopControl(environment, steps_per_episode=steps_per_episode)
        return environment, loop_controls

    def create_training_data(self, train_observations, val_observations):
        # Training data from one week
        train_features = torch.tensor(np.stack([obs['features'] for obs in train_observations]), dtype=torch.float32)
        train_actions = torch.tensor([obs['action'] for obs in train_observations], dtype=torch.long)

        # Validation data from different week
        val_features = torch.tensor(np.stack([obs['features'] for obs in val_observations]), dtype=torch.float32)
        val_actions = torch.tensor([obs['action'] for obs in val_observations], dtype=torch.long)

        # Create dataloaders
        train_dataset = TensorDataset(train_features, train_actions)
        val_dataset = TensorDataset(val_features, val_actions)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        return train_loader, val_loader

    def analyze_data_distribution(self, data):
        # Count actions
        actions = [obs['action'] for obs in data]
        unique, counts = np.unique(actions, return_counts=True)
        for value, count in zip(unique, counts):
            print(f"Action {value}: {count}")

    def run_episode(self, simulation_parameters: SimulationParameters,
                    charging_check_strategy,
                    print_freq=0, stop_condition=False,
                    log_dir='', steps_per_episode=None,
                    testing=True,
                    action_converters=None,
                    output_converter: FeatureConverterCharging = None):
        env, loop_controls = self._init_run_loop(
            simulation_parameters, log_dir, action_converters, steps_per_episode)
        parametrization_failure = False
        start = time.time()
        if output_converter:
            output_converter.reset()
        self.data = []
        while not loop_controls.done:
            decision_mode = env.core_env.decision_mode
            if decision_mode == "charging_check" or decision_mode == "charging":
                prev_event = env.core_env.previous_event
                if output_converter:
                    observations = output_converter.modify_state(loop_controls.state)
                if isinstance(charging_check_strategy, ChargingStrategy):
                    action = charging_check_strategy.get_action(loop_controls.state,
                                                                agv_id=prev_event.agv.id)
                else:
                    agv = loop_controls.state.agv_manager.agv_index[prev_event.agv.id]
                    if agv.battery <= 20:
                        action = 1
                    else:
                        action = charging_check_strategy.predict(observations)
                if output_converter:
                    self.data.append({"step": loop_controls.n_decisions, "features": observations, "action": action})
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
        if not testing:
            return parametrization_failure
        else:
            return loop_controls.state

    def test_minislap(self):
        params = SimulationParameters(
            use_case="minislap",
            use_case_n_partitions=1,
            use_case_partition_to_use=0,
            n_agvs=3,
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
            battery_capacity=10
        )

        final_state: State = run_episode(simulation_parameters=params,
                    storage_strategy=ClosestOpenLocation(very_greedy=False),
                    charging_strategy=FixedChargePolicy(100),
                    print_freq=100000, warm_start=False,
                    log_dir='./logs/tests/no_bc_single_pt/',
                    charging_check_strategy=LowTHChargePolicy(20),
                    testing=True)

    def test_env_no_battery_constraints_single_pt(self):
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            partition_by_week=True,
            use_case_partition_to_use=2,
            n_agvs=35,
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
            battery_capacity=80*1000
        )

        final_state: State = run_episode(simulation_parameters=params,
                    storage_strategy=ClosestOpenLocation(very_greedy=False),
                    charging_strategy=FixedChargePolicy(70),
                    print_freq=100000, warm_start=False,
                    log_dir='./logs/tests/no_bc_single_pt/',
                    charging_check_strategy=LowTHChargePolicy(20),
                    testing=True)
        assert final_state.trackers.average_service_time == 426.0095478607154

    # def test_env_no_battery_constraints(self):
    #     params = SimulationParameters(
    #         use_case="wepastacks_bm",
    #         use_case_n_partitions=1,
    #         use_case_partition_to_use=0,
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
    #         charging_thresholds=[40, 50, 60, 70, 80],
    #         battery_capacity=80*1000
    #     )
    #
    #
    #     final_state = run_episode(simulation_parameters=params,
    #                 storage_strategy=ClosestOpenLocation(very_greedy=False),
    #                 charging_strategy=FixedChargePolicy(70),
    #                 print_freq=100000, warm_start=False,
    #                 log_dir='./logs/tests/no_bc_single_pt/',
    #                 charging_check_strategy=LowTHChargePolicy(20), testing=True)

    def test_cs_required(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        use_case_base = "wepastacks"
        use_case = "wepastacks_bm"
        layout_path_base = get_layout_path(use_case_base)
        layout_path_present = get_layout_path(use_case)
        max_cs = 0
        # When saving/loading the model:
        for pt in range(14):
            n_cs = 0
            constraints_breached = True
            while constraints_breached:
                n_cs += 1
                if n_cs > max_cs:
                    max_cs = n_cs
                print(f"max cs required: {max_cs}")
                layout_present = pd.read_csv(layout_path_present, header=None, delimiter=",")
                n_cs_present = count_charging_stations(layout_present)
                if n_cs_present != n_cs:
                    delete_prec_dima(BASE_DIR, use_case)
                    layout_base = pd.read_csv(layout_path_base, header=None, delimiter=",")
                    layout_new = pd.DataFrame()
                    layout_new = gen_charging_stations(layout_base, n_cs)
                    layout_new.to_csv(layout_path_present,
                                      header=None, index=False)
                params = SimulationParameters(
                        use_case="wepastacks_bm",
                        use_case_n_partitions=20,
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
                        battery_capacity=52,
                        charge_during_breaks=False
                    )

                constraints_breached = run_episode(simulation_parameters=params,
                                storage_strategy=ClosestOpenLocation(very_greedy=False),
                                charging_strategy=FixedChargePolicy(40),
                                print_freq=1000, warm_start=False,
                                log_dir='./logs/tests/cs_required/',
                                charging_check_strategy=LowTHChargePolicy(20), testing=False,
                                stop_condition=True)
        print("max cs overall: ", max_cs)

    def test_amr_required(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        use_case_base = "wepastacks"
        use_case = "wepastacks_bm"
        layout_path_base = get_layout_path(use_case_base)
        layout_path_present = get_layout_path(use_case)
        max_amr = 0
        # When saving/loading the model:
        for pt in range(14):
            n_amr = 1
            constraints_breached = True
            while constraints_breached:
                n_amr += 1
                if n_amr > max_amr:
                    max_amr = n_amr
                print(f"max amr required for week {pt}: {max_amr}")
                params = SimulationParameters(
                        use_case="wepastacks_bm",
                        use_case_n_partitions=20,
                        use_case_partition_to_use=pt,
                        partition_by_week=True,
                        n_agvs=n_amr,
                        n_cs=0,
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
                        battery_capacity=52,
                        charge_during_breaks=False
                    )

                constraints_breached = run_episode(simulation_parameters=params,
                                storage_strategy=ClosestOpenLocation(very_greedy=False),
                                charging_strategy=FixedChargePolicy(100),
                                print_freq=1000, warm_start=False,
                                log_dir='./logs/tests/amr_required/',
                                charging_check_strategy=OpportunityChargePolicy(), testing=False,
                                stop_condition=True)
        print("max cs overall: ", max_amr)

    def test_cs_amr_required(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        use_case_base = "wepastacks"
        use_case = "wepastacks_bm"
        layout_path_base = get_layout_path(use_case_base)
        layout_path_present = get_layout_path(use_case)
        max_cs = 3
        max_amr = 0
        # When saving/loading the model:
        for pt in range(14):
            n_cs = max_cs
            n_amr = 33
            constraints_breached = True
            while constraints_breached:
                n_amr += 1
                if n_amr > 40:
                    n_cs += 1
                    n_amr = 1
                if n_amr > max_amr:
                    max_amr = n_amr
                if n_cs > max_cs:
                    max_cs = n_cs
                print(f"max cs/ amr required in week {pt}: {max_cs}/ {max_amr}")
                layout_present = pd.read_csv(layout_path_present, header=None, delimiter=",")
                n_cs_present = count_charging_stations(layout_present)
                if n_cs_present != n_cs:
                    delete_prec_dima(BASE_DIR, use_case)
                    layout_base = pd.read_csv(layout_path_base, header=None, delimiter=",")
                    layout_new = pd.DataFrame()
                    layout_new = gen_charging_stations(layout_base, n_cs)
                    layout_new.to_csv(layout_path_present,
                                      header=None, index=False)
                params = SimulationParameters(
                    use_case="wepastacks_bm",
                    use_case_n_partitions=20,
                    use_case_partition_to_use=pt,
                    partition_by_week=True,
                    n_agvs=n_amr,
                    n_cs=n_cs,
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
                    battery_capacity=52,
                    charge_during_breaks=False
                )

                constraints_breached = run_episode(simulation_parameters=params,
                                                   storage_strategy=ClosestOpenLocation(very_greedy=False),
                                                   charging_strategy=FixedChargePolicy(100),
                                                   print_freq=1000, warm_start=False,
                                                   log_dir=f'./logs/tests/cs_required/',
                                                   charging_check_strategy=OpportunityChargePolicy(), testing=False,
                                                   stop_condition=True)
            print(f"week {pt}: {n_cs} CS, {n_amr} AMR")
        print("max cs/ amr overall: ", max_cs, max_amr)
    # def test_env_battery_constraints_single_pt(self):
    #     params = SimulationParameters(
    #         use_case="wepastacks_bm",
    #         use_case_n_partitions=20,
    #         use_case_partition_to_use=0,
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
    #         charging_thresholds=[40, 50, 60, 70, 80],
    #         battery_capacity=80
    #     )
    #
    #
    #     final_state = run_episode(simulation_parameters=params,
    #                 storage_strategy=ClosestOpenLocation(very_greedy=False),
    #                 charging_strategy=FixedChargePolicy(70),
    #                 print_freq=100000, warm_start=False,
    #                 log_dir='./logs/tests/bc_single_pt/',
    #                 charging_check_strategy=LowTHChargePolicy(20), testing=True)
    #     assert final_state.trackers.average_service_time == 463.81978959254207
    #
    # def test_partitioning(self):
    #     n_orders_base = 411830
    #     partition_sizes = [1, 5, 10, 20, 40]
    #     for size in partition_sizes:
    #         partitions_path = get_partitions_path("wepastacks_bm")
    #         delete_partitions_data(partitions_path)
    #         params = SimulationParameters(
    #             use_case="wepastacks_bm",
    #             use_case_n_partitions=size,
    #             use_case_partition_to_use=0,
    #             n_agvs=40,
    #             generate_orders=False,
    #             verbose=False,
    #             resetting=False,
    #             initial_pallets_storage_strategy=ConstantTimeGreedyPolicy(),
    #             pure_lanes=True,
    #             n_levels=3,
    #             # https://logisticsinside.eu/speed-of-warehouse-trucks/
    #             agv_speed=2,
    #             unit_distance=1.4,
    #             pallet_shift_penalty_factor=20,  # in seconds
    #             compute_feature_trackers=True,
    #             charging_thresholds=[40, 50, 60, 70, 80],
    #             battery_capacity=80
    #         )
    #         env: SlapEnv = get_episode_env(
    #             sim_parameters=params,
    #             log_frequency=1000,
    #             nr_zones=3, log_dir='./logs/tests/partitioning/')
    #         assert env.core_env.orders.n_orders == floor(n_orders_base / size)

    def test_go_charging_combined(self):
        # Charging Action from ChargingPolicy gets overwritten by CombinedChargingPolicy
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=4,
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
            battery_capacity=80
        )

        final_state: State = run_episode(simulation_parameters=params,
                                         storage_strategy=ClosestOpenLocation(very_greedy=False),
                                         charging_strategy=FixedChargePolicy(100),
                                         print_freq=100000, warm_start=False,
                                         log_dir='./logs/tests/no_bc_single_pt/',
                                         charging_check_strategy=OpportunityChargePolicy(),
                                         testing=True)

    def test_opportunity_charging(self):
        # Charging Action from ChargingPolicy gets overwritten by CombinedChargingPolicy
        feature_list = ["n_depleted_agvs", "free_agv", "avg_battery", "utilization",
                        "queue_len_charging_station", "global_fill_level",
                        "curr_agv_battery", "dist_to_cs",
                        "queue_len_retrieval_orders", "queue_len_delivery_orders",
                        "hour_sin", "hour_cos", "day_of_week", "free_cs_available", "avg_entropy"]
        output_converter = FeatureConverterCharging(feature_list=feature_list,
                                                    reward_setting=19,
                                                    decision_mode="charging_check")
        for i in range(14):
            print(f"Week {i}")
            params = SimulationParameters(
                use_case="wepastacks_bm",
                use_case_n_partitions=20,
                use_case_partition_to_use=i,
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
                battery_capacity=52,
                charge_during_breaks=False
            )


            final_state: State = self.run_episode(simulation_parameters=params,
                                             print_freq=1000,
                                             log_dir='./logs/tests/partitioning/go_charging',
                                             charging_check_strategy=OpportunityChargePolicy(),
                                             testing=True,
                                             action_converters=[BatchFIFO(),
                                                                ClosestOpenLocation(very_greedy=False),
                                                                FixedChargePolicy(100)],
                                             steps_per_episode=None,
                                             output_converter=output_converter)
        train_obs = self.data
        self.data = []
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=1,
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
            battery_capacity=52,
            charge_during_breaks=False
        )
        final_state: State = self.run_episode(simulation_parameters=params,
                                              print_freq=1000,
                                              log_dir='./logs/tests/partitioning/go_charging',
                                              charging_check_strategy=OpportunityChargePolicy(),
                                              testing=True,
                                              action_converters=[BatchFIFO(),
                                                                 ClosestOpenLocation(very_greedy=False),
                                                                 FixedChargePolicy(100)],
                                              steps_per_episode=None,
                                              output_converter=output_converter)
        val_obs = self.data
        print("Training Data:")
        self.analyze_data_distribution(train_obs)
        print("\nValidation Data:")
        self.analyze_data_distribution(val_obs)
        train_loader, val_loader = self.create_training_data(train_obs, val_obs)
        model = train_model(train_loader, val_loader, input_dim=train_loader.dataset[0][0].shape[0])
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=1,
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
            battery_capacity=52,
            charge_during_breaks=False
        )
        final_state: State = self.run_episode(simulation_parameters=params,
                                              print_freq=1000,
                                              log_dir='./logs/tests/partitioning/go_charging',
                                              charging_check_strategy=model,
                                              testing=True,
                                              action_converters=[BatchFIFO(),
                                                                 ClosestOpenLocation(very_greedy=False),
                                                                 FixedChargePolicy(100)],
                                              steps_per_episode=None,
                                              output_converter=output_converter)
        # assert final_state.trackers.average_service_time == 463.81978959254207

    def test_charge_during_breaks(self):
        # Charging Action from ChargingPolicy gets overwritten by CombinedChargingPolicy
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=2,
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
            battery_capacity=52,
            charge_during_breaks=False
        )

        final_state: State = self.run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/go_charging',
                                         charging_check_strategy=CombinedChargingPolicy(20, 40),
                                         testing=True,
                                         action_converters=[BatchFIFO(),
                                                            ClosestOpenLocation(very_greedy=False),
                                                            FixedChargePolicy(70)],
                                         steps_per_episode=None)

        # assert final_state.trackers.average_service_time != 463.81978959254207

    def test_charge_during_breaks_week(self):
        # Charging Action from ChargingPolicy gets overwritten by CombinedChargingPolicy
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=4,
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
            battery_capacity=80,
            charge_during_breaks=True,
            partition_by_week=True
        )

        final_state: State = self.run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/go_charging',
                                         charging_check_strategy=CombinedChargingPolicy(20, 70),
                                         testing=True,
                                         action_converters=[BatchFIFO(),
                                                            ClosestOpenLocation(very_greedy=False),
                                                            FixedChargePolicy(70)],
                                         steps_per_episode=None)

        assert final_state.trackers.average_service_time != 463.81978959254207

    def test_charging_week(self):
        # Charging Action from ChargingPolicy gets overwritten by CombinedChargingPolicy
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=4,
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
            battery_capacity=80,
            partition_by_week=True
        )

        final_state: State = self.run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/go_charging',
                                         charging_check_strategy=CombinedChargingPolicy(20, 70),
                                         testing=True,
                                         action_converters=[BatchFIFO(),
                                                            ClosestOpenLocation(very_greedy=False),
                                                            FixedChargePolicy(70)],
                                         steps_per_episode=None)

        assert final_state.trackers.average_service_time != 463.81978959254207

    def test_go_charging(self):
        # Step handles the charge or not charge decision (binary). Charging duration is fixed to upper th
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
            charging_thresholds=[40, 50, 60, 70, 80],
            battery_capacity=80
        )

        final_state: State = self.run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/go_charging',
                                         charging_check_strategy=LowTHChargePolicy(20),
                                         testing=True,
                                         action_converters=[BatchFIFO(),
                                                            ClosestOpenLocation(very_greedy=False),
                                                            FixedChargePolicy(70)],
                                         steps_per_episode=None)
        assert final_state.trackers.average_service_time == 463.81978959254207

    def test_charging(self):
        # Step handles the charging duration. Go Charging is fixed to lower th
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
            charging_thresholds=[40, 50, 60, 70, 80],
            battery_capacity=80
        )

        final_state: State = self.run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/charging',
                                         charging_check_strategy=FixedChargePolicy(70),
                                         testing=True,
                                         steps_per_episode=120,
                                         action_converters=[BatchFIFO(),
                                                            ClosestOpenLocation(very_greedy=False),
                                                            LowTHChargePolicy(20)
                                                            ])
        assert final_state.trackers.average_service_time == 463.81978959254207

    def test_order_generator(self):
        # Step handles the charging duration. Go Charging is fixed to lower th
        params = SimulationParameters(
            use_case="wepastacks_bm",
            use_case_n_partitions=20,
            use_case_partition_to_use=0,
            n_agvs=5,
            generate_orders=True,
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
            battery_capacity=80,
            n_rows=20,
            n_columns=20,
            n_sources=5,
            n_sinks=5,
            desired_fill_level=0.3
        )

        final_state: State = self.run_episode(simulation_parameters=params,
                                         print_freq=1000,
                                         log_dir='./logs/tests/partitioning/charging',
                                         charging_check_strategy=FixedChargePolicy(30),
                                         testing=True,
                                         steps_per_episode=120,
                                         action_converters=[BatchFIFO(),
                                                            ClosestOpenLocation(very_greedy=False),
                                                            LowTHChargePolicy(20)
                                                            ])
        assert final_state.trackers.average_service_time != 463.81978959254207

    def test_dummy_env(self):
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
            charging_thresholds=[40, 50, 60, 70, 80],
            battery_capacity=80
        )

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

        def mask_fn(env):
            """Get action mask for environment"""
            if hasattr(env, 'valid_action_mask'):
                return env.valid_action_mask()
            if hasattr(env, 'env'):
                return mask_fn(env.env)
            if hasattr(env, 'envs'):
                # For vectorized environments, return mask for first env
                return mask_fn(env.envs[0])
            raise ValueError("Environment doesn't have valid_action_mask method")

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
                env = ActionMasker(env, mask_fn)
                return env

            return _init

        # Create vectorized environment with proper seeding
        n_envs = 4
        #random_seeds = np.random.randint(0, 100000, size=n_envs)

        # Create vectorized environment using DummyVectorEnv instead of AsyncVectorEnv
        vec_env = DummyVecEnv([
            make_env(
                params,
                log_frequency=1000,
                nr_zones=3,
                logfile_name='PPO_test',
                log_dir='./logs/tests/partitioning/charging',
                partitions=[pt],
                reward_setting=1,
            ) for pt in [0, 2]
        ])
        vec_env = VecMonitor(vec_env)
        vec_env = VecNormalize(
                    vec_env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_reward=10.0,
                    gamma=0.99,
                )
        # Initialize PPO with the vectorized environment
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            verbose=1,
            tensorboard_log="./dqn_charging_tensorboard/",
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

        # Test the environment setup
        obs = vec_env.reset()
        try:
            # Verify action masking works
            masks = vec_env.env_method('action_masks')
            print("Action masks verified:", [m.shape for m in masks])
        except Exception as e:
            print("Error testing masks:", e)
        model.learn(
            total_timesteps=50000,
            progress_bar=False,
            log_interval=1,
            tb_log_name=f"PPO_test_parallel"
        )
        # eval_env = ActionMasker(eval_env, mask_fn)
        # vec_env = AsyncVectorEnv([lambda i=i: get_env(
        #     sim_parameters=sim_params,
        #     log_frequency=1000,
        #     nr_zones=3,
        #     log_dir=cfg.experiment.log_dir,
        #     logfile_name=f"{cfg.model.agent.name}_{cfg.experiment.id}_{i}",  # Ensure unique logfile name
        #     reward_setting=cfg.task.task.reward_setting,
        #     partitions=[cfg.experiment.t_pt],
        #     cfg=cfg
        # ) for i in range(3)])
    def test_sub_proc_env(self):
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
            charging_thresholds=[40, 50, 60, 70, 80],
            battery_capacity=80
        )

        def get_rl_env(sim_parameters: SimulationParameters,
                       log_frequency: int,
                       nr_zones: int,
                       logfile_name: str,
                       log_dir: str,
                       partitions=None,
                       reward_setting=21,
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

        # Create vectorized environment with proper seeding
        n_envs = 4
        # random_seeds = np.random.randint(0, 100000, size=n_envs)

        # Create vectorized environment using DummyVectorEnv instead of AsyncVectorEnv
        vec_env = SubprocVecEnv([
            make_env(
                params,
                log_frequency=1000,
                nr_zones=3,
                logfile_name='PPO_test',
                log_dir='./logs/tests/partitioning/charging',
                partitions=[2],
                reward_setting=21,
            ) for _ in range(4)
        ])
        vec_env = VecMonitor(vec_env)
        # Initialize PPO with the vectorized environment
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            verbose=1,
            tensorboard_log="./dqn_charging_tensorboard/",
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

        # Test the environment setup
        obs = vec_env.reset()
        try:
            # Verify action masking works
            masks = vec_env.env_method('action_masks')
            print("Action masks verified:", [m.shape for m in masks])
        except Exception as e:
            print("Error testing masks:", e)
        model.learn(
            total_timesteps=100000,
            progress_bar=False,
            log_interval=1,
            tb_log_name=f"PPO_test_parallel"
        )

    def test_async_env(self):
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
            charging_thresholds=[40, 50, 60, 70, 80],
            battery_capacity=80
        )

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

        def mask_fn(env):
            """Get action mask for environment"""
            if hasattr(env, 'valid_action_mask'):
                return env.valid_action_mask()
            if hasattr(env, 'env'):
                return mask_fn(env.env)
            if hasattr(env, 'envs'):
                # For vectorized environments, return mask for first env
                return mask_fn(env.envs[0])
            raise ValueError("Environment doesn't have valid_action_mask method")

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
                env = ActionMasker(env, mask_fn)
                return env

            return _init

        # Create vectorized environment with proper seeding
        n_envs = 4
        #random_seeds = np.random.randint(0, 100000, size=n_envs)

        # Create vectorized environment using DummyVectorEnv instead of AsyncVectorEnv
        vec_env = AsyncVectorEnv([
            make_env(
                params,
                log_frequency=1000,
                nr_zones=3,
                logfile_name='PPO_test',
                log_dir='./logs/tests/partitioning/charging',
                partitions=[pt],
                reward_setting=1,
            ) for pt in [0, 2, 4, 6, 8, 10]
        ])

        # Initialize PPO with the vectorized environment
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            verbose=1,
            tensorboard_log="./dqn_charging_tensorboard/",
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )

        # Test the environment setup
        obs = vec_env.reset()
        try:
            # Verify action masking works
            masks = vec_env.env_method('action_masks')
            print("Action masks verified:", [m.shape for m in masks])
        except Exception as e:
            print("Error testing masks:", e)
        model.learn(
            total_timesteps=1000,
            progress_bar=False,
            log_interval=1,
            tb_log_name=f"PPO_test_parallel"
        )
    # def test_partition_cycling(self):
    #     partitions_path = get_partitions_path("wepastacks_bm")
    #     delete_partitions_data(partitions_path)
    #     params = SimulationParameters(
    #         use_case="wepastacks_bm",
    #         use_case_n_partitions=20,
    #         use_case_partition_to_use=1,
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
    #         charging_thresholds=[40, 50, 60, 70, 80],
    #         battery_capacity=80
    #     )
    #     partitions = [0, 13, 17, 19]
    #
    #     env: SlapEnv = get_episode_env(
    #         sim_parameters=params,
    #         log_frequency=1000,
    #         nr_zones=3, log_dir='./logs/tests/partitioning/',
    #         partitions=partitions)
    #     for pt in range(len(partitions)):
    #         partitions_path = get_partitions_path("wepastacks_bm")
    #         if pt < len(partitions):
    #             idx = pt + 1
    #         else:
    #             idx = 0
    #         with open(join(partitions_path, f"{partitions[idx]}_partition_fill_lvl.json")) as json_file:
    #             initial_fill_json = json.load(json_file)
    #         skus_ini = defaultdict(int)
    #         all_skus = set(skus_ini.keys())
    #         for sku, amount in initial_fill_json.items():
    #             skus_ini[int(sku)] = amount
    #             all_skus.add(int(sku))
    #         env.reset()
    #         print(sum(skus_ini.values()))
    #         assert sum(env.core_env.orders.initial_pallets_sku_counts.values()) == sum(skus_ini.values())

    # def test_env_multi_sources_sinks(self):
    #     """tests slap env with 3 sources, 3 sinks, and a small order list"""
    #     environment_parameters,seeds, log_path = get_use_case_parameters()
    #     environment_parameters['verbose'] = False
    #     environment_parameters['order_list'] = [("retrieval", 1, 0, 0),
    #                                             ("delivery", 2, 100, 1),
    #                                             ("delivery", 1, 300, 2),
    #                                             ("retrieval", 2, 400, 1),
    #                                             ("retrieval", 1, 500, 2),
    #                                             ("delivery", 1, 600, 0)]
    #     environment_parameters['n_levels'] = 2
    #     environment_parameters['n_sources'] = 4
    #     environment_parameters['n_sinks'] = 3
    #     log_path = ''
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     if env.verbose:
    #         print(env.state)
    #         print(env.state.trackers)
    #         print("finished all orders")
    #     self.assertTrue(done)
    #
    #
    # def test_env_short_no_overlaps(self):
    #     """tests slap env with small order list and no overlaps"""
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     environment_parameters['verbose'] = False
    #     log_path = ''
    #     environment_parameters['order_list'] = [("retrieval", 1, 0, 0, 1),
    #                                             ("delivery", 2, 100, 0, 1),
    #                                             ("delivery", 1, 300, 0, 1),
    #                                             ("retrieval", 2, 400, 0, 1),
    #                                             ("retrieval", 1, 500, 0, 1),
    #                                             ("delivery", 1, 600, 0, 1)]
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     if env.verbose:
    #         print(env.state)
    #         print(env.state.trackers)
    #         print("finished all orders")
    #     self.assertTrue(done)
    #
    # def test_env_short_with_overlaps(self):
    #     """tests slap env with small order list and with overlaps"""
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     log_path = ''
    #
    #     environment_parameters['order_list'] = [("retrieval", 1, 0, 0),
    #                                             ("delivery", 2, 10, 0),
    #                                             ("delivery", 1, 15, 0),
    #                                             ("retrieval", 2, 20, 0),
    #                                             ("retrieval", 1, 25, 0),
    #                                             ("delivery", 1, 30, 0)]
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     if env.verbose:
    #         print(env.state)
    #         print("finished all orders")
    #     self.assertTrue(done)
    #
    # def test_env_long(self):
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     environment_parameters['verbose'] = False
    #     log_path=''
    #     environment_parameters['generate_orders'] = True
    #     environment_parameters['order_list'] = None
    #     environment_parameters['n_levels'] = 2
    #     environment_parameters['n_rows'] = 12
    #     environment_parameters['n_columns'] = 12
    #     environment_parameters['n_orders'] = 100
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     if environment_parameters['verbose']:
    #         print(env.state)
    #         print("finished successfully")
    #     self.assertTrue(done)
    #
    # def test_many_agvs(self):
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     log_path = ''
    #
    #     environment_parameters['verbose'] = True
    #     environment_parameters['generate_orders'] = True
    #     environment_parameters['desired_fill_level']= 0.4
    #     environment_parameters['order_list'] = None
    #     environment_parameters['n_levels'] = 2
    #     environment_parameters['n_rows'] = 10
    #     environment_parameters['n_agvs'] = 1
    #     environment_parameters['n_columns'] = 15
    #     environment_parameters['n_orders'] = 50
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #         print_3d_np(env.state.S)
    #     if environment_parameters['verbose']:
    #         print(env.state)
    #         print("finished successfully")
    #     self.assertTrue(done)
    #
    # def test_env_initial_random_pallets_no_storage_strategy(self):
    #     dictionary = {1: 4, 2: 6, 3: 5}
    #
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     log_path = ''
    #
    #     environment_parameters['verbose'] = False
    #     environment_parameters['generate_orders'] = True
    #     environment_parameters['order_list'] = None
    #     environment_parameters['n_levels'] = 1
    #     environment_parameters['n_rows'] = 12
    #     environment_parameters['n_columns'] = 12
    #     environment_parameters['n_orders'] = 10
    #     environment_parameters['initial_pallets_sku_counts'] = dictionary.copy()
    #     environment_parameters['n_skus'] = 3
    #
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     assertion_dictionary = dictionary.copy()
    #     for sku in assertion_dictionary:
    #         sku_count = len(np.argwhere(env.storage_matrix_history[1] == sku))
    #         assertion_dictionary[sku] = sku_count
    #     # if environment_parameters['verbose'] is True:
    #     # print_3d_np(env.storage_matrix_history[1])
    #     # print(assertion_dictionary)
    #     self.assertDictEqual(dictionary, assertion_dictionary)
    #
    # def test_env_initial_random_pallets_with_dict_and_storage_strategy(self):
    #     dictionary = {1: 4, 2: 6, 3: 5}
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     log_path = ''
    #
    #     environment_parameters['verbose'] = False
    #     environment_parameters['generate_orders'] = True
    #     environment_parameters['order_list'] = None
    #     environment_parameters['n_levels'] = 4
    #     environment_parameters['n_rows'] = 12
    #     environment_parameters['n_columns'] = 12
    #     environment_parameters['n_orders'] = 10
    #     environment_parameters['initial_pallets_sku_counts'] = dictionary.copy()
    #     environment_parameters['n_skus'] = 3
    #     environment_parameters['initial_pallets_storage_strategy'] = FurthestOpenLocation()
    #
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     assertion_dictionary = dictionary.copy()
    #     for sku in assertion_dictionary:
    #         sku_count = len(np.argwhere(env.storage_matrix_history[1] == sku))
    #         assertion_dictionary[sku] = sku_count
    #     # if environment_parameters['verbose'] is True:
    #     # print_3d_np(env.storage_matrix_history[1])
    #     # print(assertion_dictionary)
    #     self.assertDictEqual(dictionary, assertion_dictionary)
    #
    # def test_one_delivery_order(self):
    #     """tests slap env with only one delivery order"""
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     log_path = ''
    #
    #     environment_parameters['verbose'] = False
    #     environment_parameters['order_list'] = [("delivery", 1, 0, 0)]
    #     environment_parameters['n_agvs'] = 1
    #     environment_parameters['n_skus'] = 1
    #     environment_parameters['n_levels'] = 1
    #     environment_parameters['initial_pallets_sku_counts'] = {1: 0}
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     if env.verbose:
    #         print(env.state)
    #         print(env.state.trackers)
    #         print("finished all orders")
    #
    #     with self.subTest():
    #         self.assertTrue(done)
    #     with self.subTest():
    #         self.assertEqual("delivery", env.decision_mode)
    #     with self.subTest():
    #         self.assertEqual("delivery", env.state.current_order)
    #     with self.subTest():
    #         self.assertEqual(1, env.previous_event.SKU)
    #     with self.subTest():
    #         self.assertEqual(1, env.state.current_sku)
    #
    # def test_one_retrieval_order(self):
    #     """tests slap env with only one retrieval order"""
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     log_path = ''
    #
    #     environment_parameters['verbose'] = False
    #     environment_parameters['order_list'] = [("retrieval", 1, 0, 0)]
    #     environment_parameters['n_agvs'] = 1
    #     environment_parameters['n_skus'] = 1
    #     environment_parameters['n_levels'] = 1
    #     environment_parameters['initial_pallets_sku_counts'] = {1: 1}
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     if env.verbose:
    #         print(env.state)
    #         print(env.state.trackers)
    #         print("finished all orders")
    #     with self.subTest():
    #         self.assertTrue(done)
    #     with self.subTest():
    #         self.assertEqual("retrieval", env.decision_mode)
    #     with self.subTest():
    #         self.assertEqual("retrieval", env.state.current_order)
    #     with self.subTest():
    #         self.assertEqual(1, env.previous_event.SKU)
    #     with self.subTest():
    #         self.assertEqual(1, env.state.current_sku)
    #
    # def test_not_refilling_warehouse(self):
    #     """tests slap env with small order list and no overlaps"""
    #     environment_parameters, seeds, log_path = get_use_case_parameters()
    #     environment_parameters['verbose'] = False
    #     log_path = ''
    #     environment_parameters['n_levels'] = 1
    #     environment_parameters['order_list'] = [("delivery", 2, 100, 0, 1),
    #                                             ("delivery", 1, 300, 0, 1),
    #                                             ("delivery", 1, 600, 0, 1)]
    #     parameters = Input(environment_parameters, log_path, seeds)
    #     env = SlapCore(parameters)
    #     env.reset()
    #     state = env.get_state()
    #     done = False
    #     while not done:
    #         legal_actions = env.get_legal_actions()
    #         state, done = env.step(random.choice(legal_actions))
    #     storage_matrix = np.copy(env.state.S)
    #     # print_3d_np(storage_matrix)
    #     env.reset(False)
    #     new_storage_matrix = np.copy(env.state.S)
    #     # test=env.state_stack
    #     # print_3d_np(new_storage_matrix)
    #     self.assertEqual(storage_matrix.tolist(), new_storage_matrix.tolist())