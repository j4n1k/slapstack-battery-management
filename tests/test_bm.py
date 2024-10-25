import json
import time
from collections import defaultdict
from math import floor
from os.path import join
from unittest import TestCase

from experiments.experiment_commons import run_episode, get_episode_env, get_partitions_path, delete_partitions_data, \
    ExperimentLogger, LoopControl
from slapstack import SlapCore, SlapEnv
import random
import numpy as np

from slapstack.core_state import State
from slapstack.helpers import print_3d_np
from slapstack.interface_input import Input
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.charging_policies import FixedChargePolicy, LowTHChargePolicy
from slapstack_controls.storage_policies import ConstantTimeGreedyPolicy, ClosestOpenLocation, BatchFIFO


class TestSlapEnv(TestCase):
    def test_env_no_battery_constraints_single_pt(self):
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

    def test_env_battery_constraints_single_pt(self):
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


        final_state = run_episode(simulation_parameters=params,
                    storage_strategy=ClosestOpenLocation(very_greedy=False),
                    charging_strategy=FixedChargePolicy(70),
                    print_freq=100000, warm_start=False,
                    log_dir='./logs/tests/bc_single_pt/',
                    charging_check_strategy=LowTHChargePolicy(20), testing=True)
        assert final_state.trackers.average_service_time == 463.81978959254207

    def test_partitioning(self):
        n_orders_base = 411830
        partition_sizes = [1, 5, 10, 20, 40]
        for size in partition_sizes:
            partitions_path = get_partitions_path("wepastacks_bm")
            delete_partitions_data(partitions_path)
            params = SimulationParameters(
                use_case="wepastacks_bm",
                use_case_n_partitions=size,
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
            env: SlapEnv = get_episode_env(
                sim_parameters=params,
                log_frequency=1000,
                nr_zones=3, log_dir='./logs/tests/partitioning/')
            assert env.core_env.orders.n_orders == floor(n_orders_base / size)

    def test_go_charging(self):
        def get_episode_env(sim_parameters: SimulationParameters,
                            log_frequency: int, nr_zones: int,
                            logfile_name: str,
                            log_dir: str):
            seeds = [56513]
            return SlapEnv(
                sim_parameters, seeds,
                logger=ExperimentLogger(
                    filepath=log_dir,
                    n_steps_between_saves=log_frequency,
                    nr_zones=nr_zones,
                    logfile_name=logfile_name),
                action_converters=[BatchFIFO(),
                                   ClosestOpenLocation(very_greedy=False),
                                   FixedChargePolicy(70)])

        def _init_run_loop(simulation_parameters,
                           log_dir):
            environment: SlapEnv = get_episode_env(
                sim_parameters=simulation_parameters,
                log_frequency=1000, nr_zones=3, log_dir=log_dir,
                logfile_name="go_charging_test")
            loop_controls = LoopControl(environment, steps_per_episode=120)
            # state.state_cache.perform_sanity_check()
            return environment, loop_controls

        def run_episode(simulation_parameters: SimulationParameters,
                        charging_check_strategy,
                        print_freq=0, stop_condition=False,
                        log_dir='', steps_per_episode=None,
                        testing=True):
            env, loop_controls = _init_run_loop(
                simulation_parameters, log_dir)
            parametrization_failure = False
            start = time.time()
            while not loop_controls.done:
                if env.core_env.decision_mode == "charging_check":
                    prev_event = env.core_env.previous_event
                    action = charging_check_strategy.get_action(loop_controls.state,
                                                                agv_id=prev_event.agv.id)
                else:
                    if env.done_during_init:
                        raise ValueError("Sim ended during init")
                    raise ValueError
                output, reward, loop_controls.done, info = env.step(action)
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
                # if not loop_controls.done and stop_condition:
                #     loop_controls.done = loop_controls.stop_prematurely()
                #     if loop_controls.done:
                #         parametrization_failure = True
                #         env.core_env.logger.write_logs()
            ExperimentLogger.print_episode_info(
                charging_check_strategy.name, start, loop_controls.n_decisions,
                loop_controls.state)
            if not testing:
                return parametrization_failure
            else:
                return loop_controls.state

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

        final_state: State = run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/go_charging',
                                         charging_check_strategy=LowTHChargePolicy(20),
                                         testing=True)
        assert final_state.trackers.average_service_time == 463.81978959254207

    def test_charging(self):
        def get_episode_env(sim_parameters: SimulationParameters,
                            log_frequency: int, nr_zones: int,
                            logfile_name: str,
                            log_dir: str):
            seeds = [56513]
            return SlapEnv(
                sim_parameters, seeds,
                logger=ExperimentLogger(
                    filepath=log_dir,
                    n_steps_between_saves=log_frequency,
                    nr_zones=nr_zones,
                    logfile_name=logfile_name),
                action_converters=[BatchFIFO(),
                                   ClosestOpenLocation(very_greedy=False),
                                   ])

        def _init_run_loop(simulation_parameters,
                           log_dir):
            environment: SlapEnv = get_episode_env(
                sim_parameters=simulation_parameters,
                log_frequency=1000, nr_zones=3, log_dir=log_dir,
                logfile_name="go_charging_test")
            loop_controls = LoopControl(environment, steps_per_episode=120)
            # state.state_cache.perform_sanity_check()
            return environment, loop_controls

        def run_episode(simulation_parameters: SimulationParameters,
                        charging_check_strategy,
                        print_freq=0, stop_condition=False,
                        log_dir='', steps_per_episode=None,
                        testing=True):
            env, loop_controls = _init_run_loop(
                simulation_parameters, log_dir)
            parametrization_failure = False
            start = time.time()
            while not loop_controls.done:
                if env.core_env.decision_mode == "charging":
                    prev_event = env.core_env.previous_event
                    action = charging_check_strategy.get_action(loop_controls.state,
                                                                agv_id=prev_event.agv.id)
                else:
                    if env.done_during_init:
                        raise ValueError("Sim ended during init")
                    raise ValueError
                output, reward, loop_controls.done, info = env.step(action)
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
                # if not loop_controls.done and stop_condition:
                #     loop_controls.done = loop_controls.stop_prematurely()
                #     if loop_controls.done:
                #         parametrization_failure = True
                #         env.core_env.logger.write_logs()
            ExperimentLogger.print_episode_info(
                charging_check_strategy.name, start, loop_controls.n_decisions,
                loop_controls.state)
            if not testing:
                return parametrization_failure
            else:
                return loop_controls.state

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

        final_state: State = run_episode(simulation_parameters=params,
                                         print_freq=100000,
                                         log_dir='./logs/tests/partitioning/go_charging',
                                         charging_check_strategy=FixedChargePolicy(70),
                                         testing=True)
        assert final_state.trackers.average_service_time == 463.81978959254207

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