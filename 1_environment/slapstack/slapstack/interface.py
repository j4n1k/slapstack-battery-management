import heapq as heap
from typing import Union, List

import gymnasium as gym
from copy import deepcopy
import numpy as np

from slapstack.core_events import DeliveryFirstLeg, RetrievalFirstLeg, \
    RetrievalSecondLeg, DeliverySecondLeg
from slapstack.interface_input import Input
from slapstack.core import SlapCore
from slapstack.helpers import faster_deepcopy

from slapstack.interface_templates import SimulationParameters, SlapLogger, \
    OutputConverter, StorageStrategy


class SlapEnv(gym.Env):
    def __init__(self, environment_parameters: SimulationParameters, seeds='',
                 partitions='',
                 logger: Union[SlapLogger, str, None] = None,
                 state_converter: Union[OutputConverter, None] = None,
                 action_converters: Union[List[StorageStrategy], None] = None):
        self.last_reward = None
        self.last_action_taken = None
        if bool(seeds):
            self.__seeds_remaining = seeds[1:]
            self.__seeds_used = [seeds[0]]
            init_seed = seeds[0]
        else:
            self.__seeds_remaining = []
            self.__seeds_used = []
            init_seed = 1
        if bool(partitions):
            self.__partitions_remaining = partitions[1:]
            self.__partitions_used = [partitions[0]]
            init_partition = partitions[0]
        else:
            self.__partitions_remaining = []
            self.__partitions_used = []
            init_partition = None
        self.__env_input = Input(environment_parameters, init_seed, init_partition)
        self.__core = SlapCore(deepcopy(self.__env_input), logger)
        self.__core.reset(None)
        self.__output_converter = state_converter

        if state_converter:
            self.feature_list = state_converter.feature_list
        self.__strategy_configuration = -1
        self.__storage_strategies = None
        self.__retrieval_strategies = None
        self.__charging_strategies = None
        self.__setup_strategies(action_converters
                                if action_converters is not None else [])
        self.action_space = None
        self.observation_space = None
        self.__get_action_space()
        self.__get_observation_space()
        self.done_during_init = False
        if self.autoplay():
            _, _, done = self.__skip_fixed_decisions(False)
            if done:
                self.done_during_init = True

    def make_deterministic(self):
        travel_events = []
        for event in self.__core.events.running:
            if (isinstance(event, DeliveryFirstLeg) or
                    isinstance(event,DeliverySecondLeg) or
                    isinstance(event, RetrievalFirstLeg) or
                    isinstance(event, RetrievalSecondLeg)):
                heap.heappush(travel_events, event)
        self.__core.events.running = travel_events

    # <editor-fold desc="strategy Configuration">
    def __setup_strategies(self, selectable_strategies: list):
        """
        Splits the storages and retrieval strategies according to their
        strategy_type parameter, and initializes the strategy_configuration
        parameter defining the action space definition and action selection
        schemes.

        :param selectable_strategies: The list of strategies.
        :return: None
        """
        sto_opt, ret_opt, ch_opt = [], [], []
        for strategy in selectable_strategies:
            if strategy.type == 'delivery':
                sto_opt.append(strategy)
            elif strategy.type == 'retrieval':
                ret_opt.append(strategy)
            elif strategy.type == 'charging':
                ch_opt.append(strategy)
            else:
                raise Exception("no target mode set")
        self.__storage_strategies = np.array(sto_opt)
        self.__retrieval_strategies = np.array(ret_opt)
        self.__charging_strategies = np.array(ch_opt)
        self.__setup_strategy_config()

    def step(self, action: Union[int, np.ndarray]):
        # TODO: revisit and refactor this!!
        if self.__core.state.params.charging_thresholds:
            self.last_action_taken = action
        else:
            self.last_action_taken = action[0]
        decision_mode = self.__core.decision_mode
        direct_action = self.__transform_action(action)
        state, done = self.__core.step(direct_action)
        # assert np.ravel_multi_index(direct_action, self.__core.state.S.shape)[0:2] != (0, 0)
        # if np.ravel_multi_index(direct_action, self.__core.state.S.shape)[0:2] == (
        #         0,0):
        #     print("storage strategy:",  action)
        if self.__output_converter is None:
            # something other than RL is using this simulation
            if self.autoplay() and not done:
                state, reward, done = self.__skip_fixed_decisions(done)
            return state, None, done, {}
        legal_actions = self.get_legal_actions()
        state_repr = self.__output_converter.modify_state(self.__core.state)
        reward = self.__output_converter.calculate_reward(self.__core.state, action, legal_actions,
                                                          decision_mode)
        self.last_reward = reward
        # self.__update_strategies(direct_action)
        if self.autoplay() and not done:
            # state_repr, reward, done = self.__skip_fixed_decisions(done)
            state_repr, _, done = self.__skip_fixed_decisions(done)
        return state_repr, reward, done, False, {}

    def test_sameSKU(self):
        for aisle in self.__core.state.state_cache.lane_manager.lane_index:
            for dir in self.__core.state.state_cache.lane_manager.lane_index[aisle]:
                stacks_in_lane = self.__core.state.state_cache.lane_manager.lane_index[aisle][dir]
                stacks_in_lane = [(*stack, 0) for stack in stacks_in_lane]
                stacks_in_lane_np = list(zip(*stacks_in_lane))

                lane = self.__core.state.S[stacks_in_lane_np[0], stacks_in_lane_np[1]]

                skus = 0
                for i in range(1, self.__env_input.params.n_skus + 1):
                    skus += np.any(lane == i)

                assert skus <= 1, lane

    def seed(self, seed=None):
        self.__core.inpt.seed = seed
        self.__core.set_seed(seed)
        return [seed]

    # def seed(self, seed):
    #     self.__core.seeds['np'] = seed
    #     self.__core.seeds['random'] = seed
    #     return [self.__core.seeds['np'], self.__core.seeds['random']]

    def get_legal_actions(self):
        n_ss = self.__storage_strategies.shape[0]
        n_rs = self.__retrieval_strategies.shape[0]
        if self.__strategy_configuration in {0, 1, 3, 4}:
            return self.__core.legal_actions
        elif self.__strategy_configuration == 2:
            if self.__core.decision_mode == "delivery":
                return self.__core.legal_actions
            else:
                return [i for i in range(n_rs)]
        elif self.__strategy_configuration == 5:
            if self.__core.decision_mode == "delivery":
                return self.__core.legal_actions
            else:
                return [i for i in range(n_rs)]
        elif self.__strategy_configuration == 6:
            if self.__core.decision_mode == "delivery":
                return [i for i in range(n_ss)]
            else:
                return self.__core.legal_actions
        elif self.__strategy_configuration == 7:
            if self.__core.decision_mode == "delivery":
                return [i for i in range(n_ss)]
            else:
                return self.__core.legal_actions
        elif self.__strategy_configuration == 8:
            if self.__core.decision_mode == "delivery":
                return [i for i in range(n_ss)]
            else:
                return [i for i in range(n_rs)]

    def reset(self, seed=None, options=None):
        # self.__core = SlapCore(self.__env_input)
        # seed cycling if seeds were passed
        if ((bool(self.__seeds_remaining) or bool(self.__seeds_used)) or
                (bool(self.__partitions_remaining) or bool(self.__partitions_used))):
            if len(self.__seeds_remaining) > 0:
                seed = self.__seeds_remaining.pop(0)
                self.__seeds_used.append(seed)
            else:
                self.__seeds_remaining = self.__seeds_used[1:]
                seed = self.__seeds_used[0]
                self.__seeds_used = [seed]
            if len(self.__partitions_remaining) > 0:
                pt = self.__partitions_remaining.pop(0)
                # self.__partitions_remaining.append(pt)
                self.__partitions_used.append(pt)
            else:
                self.__partitions_remaining = self.__partitions_used[1:]
                pt = self.__partitions_used[0]
                self.__partitions_used = [pt]
            self.__env_input = Input(self.__env_input.params, seed, pt)
        else:
            self.__env_input = Input(self.__env_input.params)
        self.__core = SlapCore(self.__env_input, self.__core.logger)
        self.__core.reset(None)
        if self.autoplay():
            self.__skip_fixed_decisions(False)
        if self.__output_converter is not None:
            self.__output_converter.reset()
            return self.__output_converter.modify_state(
                self.__core.state), None
        else:
            return self.__core.state

    def __skip_fixed_decisions(self, done):
        state_repr, reward = None, 0
        while self.autoplay() and not done:
            # self.core_env.state.current_agv = self.core_env.previous_event.agv_id
            if self.__core.state.current_order == "delivery":
                action = self.__storage_strategies[0].get_action(
                    self.__core.state)
                state, done = self.__core.step(action)
                if self.__output_converter:
                    state_repr = self.__output_converter.modify_state(self.__core.state)
                    if not done:
                        legal_actions = self.__core.legal_actions
                        reward = self.__output_converter.calculate_reward(self.__core.state, action, legal_actions,
                                                                          "delivery")
                else:
                    state_repr = state
            elif self.__core.state.current_order == "retrieval":  # if current order is a retrieval order
                action = self.__retrieval_strategies[0].get_action(self.__core.state)
                state, done = self.__core.step(action)
                if self.__output_converter:
                    state_repr = self.__output_converter.modify_state(self.__core.state)
                    if not done:
                        legal_actions = self.__core.legal_actions
                        reward = self.__output_converter.calculate_reward(self.__core.state, action, legal_actions,
                                                                          "retrieval")
                else:
                    state_repr = state
            elif self.__core.state.current_order == "charging_check":
                action = self.core_env.state.agv_manager.charge_needed(False, self.core_env.previous_event.agv.id)
                state, done = self.__core.step(action)
                if self.__output_converter:
                    state_repr = self.__output_converter.modify_state(self.__core.state)
                    if not done:
                        legal_actions = self.__core.legal_actions
                        reward = self.__output_converter.calculate_reward(self.__core.state, action, legal_actions,
                                                                          "charging_check")
                else:
                    state_repr = state
            elif self.__core.state.current_order == "charging":
                action = self.__charging_strategies[0].get_action(self.__core.state,
                                                                  self.core_env.previous_event.agv.id)
                state, done = self.__core.step(action)
                if self.__output_converter:
                    state_repr = self.__output_converter.modify_state(self.__core.state)
                    if not done:
                        legal_actions = self.__core.legal_actions
                        reward = self.__output_converter.calculate_reward(self.__core.state, action, legal_actions,
                                                                          "charging")
                else:
                    state_repr = state
            # if self.__output_converter:
            #     assert state_repr.shape == (900, )
        if self.__core.decision_mode == "charging" or self.__core.decision_mode == "charging_check":
            self.current_state_repr = state_repr
            # self.core_env.state.current_agv = self.core_env.previous_event.agv.id
        return state_repr, reward, done

    def autoplay(self):
        """
        checks whether the next action can be played automatically, like
        when there is a fixed strategy (config 1, 3, 4, 5 and 7)
         """
        return ((self.__core.decision_mode == "delivery" and
                 self.__strategy_configuration in {3, 4, 5, 9}) or
                (self.__core.decision_mode == "retrieval" and
                 self.__strategy_configuration in {1, 4, 7, 9}) or
                (self.__core.decision_mode == "charging_check" and
                 self.__strategy_configuration == 4) or
                (self.__core.decision_mode == "charging" and
                 self.__strategy_configuration == 9))

    def __setup_strategy_config(self):
        """
        Initializes the strategy_configuration parameter influencing the action
        space definition and action translation to one of 11 integer values
        defined as follows:

         0: Direct storage action and direct retrieval action
         1: Direct storage action and fixed retrieval strategy
         2: Direct storage strategy and selectable retrieval strategy
         3: Fixed storage strategy and direct retrieval action
         4: Fixed storage strategy and retrieval strategy run
         5: Fixed storage strategy and selectable retrieval strategy
         6: Selectable storage strategy and direct retrieval action
         7: Selectable storage strategy and fixed retrieval strategy
         8: Selectable storage and retrieval strategies
         9: Fixed storage strategy, fixed retrieval strategy and fixed charging strategy run

        :return: None
        """

        n_ss = self.__storage_strategies.shape[0]
        n_rs = self.__retrieval_strategies.shape[0]
        n_cs = self.__charging_strategies.shape[0]
        if n_ss == 1 and n_rs == 1 and n_cs == 1:
            self.__strategy_configuration = 9
        elif n_ss == 0 and n_rs == 0:  # direct actions only
            self.__strategy_configuration = 0
        elif n_ss == 0 and n_rs == 1:
            self.__strategy_configuration = 1
        elif n_ss == 0 and n_rs > 1:
            self.__strategy_configuration = 2
        elif n_ss == 1 and n_rs == 0:
            self.__strategy_configuration = 3
        elif n_ss == 1 and n_rs == 1:
            self.__strategy_configuration = 4
        elif n_ss == 1 and n_rs > 1:
            self.__strategy_configuration = 5
        elif n_ss > 1 and n_rs == 0:
            self.__strategy_configuration = 6
        elif n_ss > 1 and n_rs == 1:
            self.__strategy_configuration = 7
        else:  # n_ss > 1 and n_rs > 1:
            self.__strategy_configuration = 8

    def __update_strategies(self, direct_action):
        if self.__storage_strategies is not None:
            for strategy in self.__storage_strategies:
                strategy.update(direct_action)
        if self.__retrieval_strategies is not None:
            for strategy in self.__retrieval_strategies:
                strategy.update(direct_action)

    def __get_action_space(self):
        """
        Initializes the action space parameter based on the
        strategy_configuration.

        :return: None
        """
        assert -1 < self.__strategy_configuration <= 9
        n_rows = self.__core.inpt.params.n_rows
        n_columns = self.__core.inpt.params.n_columns
        n_levels = self.__core.inpt.params.n_levels
        n_ss = self.__storage_strategies.shape[0]
        n_rs = self.__retrieval_strategies.shape[0]
        n_cs = self.__charging_strategies.shape[0]
        if self.__strategy_configuration in {0, 1, 3}:
            self.action_space = gym.spaces.Discrete(
                n_rows * n_columns * n_levels)
        elif self.__strategy_configuration == 2:
            self.action_space = gym.spaces.Discrete(
                n_rs + n_rows * n_columns * n_levels)
        elif self.__strategy_configuration == 4:
            if isinstance(self.__core.inpt.params.charging_thresholds, list):
                n_thresholds = len(self.__core.inpt.params.charging_thresholds)
                self.action_space = gym.spaces.Discrete(n_thresholds)
            elif isinstance(self.__core.inpt.params.charging_thresholds, tuple):
                low, high = self.__core.inpt.params.charging_thresholds
                self.action_space = gym.spaces.Box(low=low, high=high)
        elif self.__strategy_configuration == 5:
            self.action_space = gym.spaces.Discrete(n_rs)
        elif self.__strategy_configuration == 6:
            self.action_space = gym.spaces.Discrete(
                n_ss + n_rows * n_columns * n_levels)
        elif self.__strategy_configuration == 7:
            self.action_space = gym.spaces.Discrete(n_ss)
        elif self.__strategy_configuration == 9:
            # Go to charging -> binary decision
            if isinstance(self.__core.inpt.params.charging_thresholds, tuple):
                low, high = self.__core.inpt.params.charging_thresholds
                self.action_space = gym.spaces.Box(low=low, high=high)
            else:
                self.action_space = gym.spaces.discrete.Discrete(2)
        else:  # self.__strategy_configuration == 8:
            self.action_space = gym.spaces.Discrete(n_ss + n_rs)

    def __get_observation_space(self):
        """
        Initializes the observation space required by gym to a Box object as
        defined by gym.

        The observation (i.e. state) space dimension is inferred from the state
        representation returned by the state_transformer on the initial state.
        :return: None
        """
        if self.__output_converter is None:
            # something other than RL is using this simulation
            return
        state_repr = self.__output_converter.modify_state(
            self.__core.state)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=state_repr.shape)

    def __transform_action(self, agent_action):
        """
        Switches between the 9 available decision interfaces and transforms the
        agent action accordingly into an environment core compatible decision.

        :param agent_action: The action as chosen by the agent.
        :return: The action compatible with the core.
        """
        if ((self.__core.decision_mode == "charging" and
             not self.__output_converter) or
                (self.__core.decision_mode == "charging_check" and not self.__output_converter)):
            # TODO: skip if RL
            return self.__transform_a_charging_duration(agent_action)
        if self.__strategy_configuration == 0:
            # both routing and sequencing direct actions
            return self.__transform_a_direct_action_run(agent_action)
        elif self.__strategy_configuration == 1:
            return self.__transform_a_direct_storage_fixed_retrieval(
                agent_action)
        elif self.__strategy_configuration == 2:
            return self.__transform_a_direct_storage_selectable_retrieval(
                agent_action)
        elif self.__strategy_configuration == 3:
            return self.__transform_a_fixed_storage_direct_retrieval(
                agent_action)
        elif self.__strategy_configuration == 4:
            return self.__transform_a_fixed_storage_fixed_retrieval(
                agent_action)
        elif self.__strategy_configuration == 5:
            return self.__transform_a_fixed_storage_selectable_retrieval(
                agent_action)
        elif self.__strategy_configuration == 6:
            return self.__transform_a_selectable_storage_direct_retrieval(
                agent_action)
        elif self.__strategy_configuration == 7:
            return self.__transform_a_selectable_storage_fixed_retrieval(
                agent_action)
        elif self.__strategy_configuration == 8:
            return self.__transform_a_selectable_storage_selectable_retrieval(
                agent_action)
        elif self.__strategy_configuration == 9:
            return self.__transform_a_fixed_storage_fixed_retrieval(
                agent_action)

        else:  # should not be possible at this point;
            raise Exception("no strategy configured")

    def __transform_a_direct_action_run(self, agent_action):
        # config 0
        return agent_action

    def __transform_a_direct_storage_fixed_retrieval(self, agent_action):
        # config 1
        if self.__core.decision_mode == "delivery":
            return agent_action
        else:
            return self.__retrieval_strategies[0].get_action(self.__core.state)

    def __transform_a_direct_storage_selectable_retrieval(self, agent_action):
        # config 2
        if self.__core.decision_mode == "delivery":
            return agent_action
        else:
            return self.__retrieval_strategies[agent_action].get_action(self.__core.state)

    def __transform_a_fixed_storage_direct_retrieval(self, agent_action):
        # config 3
        if self.__core.decision_mode == "delivery":
            return self.__storage_strategies[0].get_action(self.__core.state)
        else:
            return agent_action

    def __transform_a_fixed_storage_fixed_retrieval(self, agent_action):
        if self.__core.decision_mode == "charging":
            if isinstance(self.__core.inpt.params.charging_thresholds, list):
                threshold = self.env_input.params.charging_thresholds[
                    agent_action]
            else:
                # threshold = self.__denormalize_action(agent_action[0])
                threshold = agent_action[0]
            agv_id = self.core_env.previous_event.agv_id
            agvm = self.core_env.state.agv_manager
            agv = self.core_env.state.agv_manager.agv_index[agv_id]
            charge_to_full = threshold - agv.battery
            charging_duration = charge_to_full / agvm.charging_rate
            # charging_duration = threshold / agvm.charging_rate
            return charging_duration
        if self.__core.decision_mode == "delivery":
            return self.__storage_strategies[0].get_action(self.__core.state)
        elif self.__core.decision_mode == "charging_check":
            # return agent_action[0].item()
            return agent_action
        else:
            return self.__retrieval_strategies[0].get_action(self.__core.state)

    def __transform_a_fixed_storage_selectable_retrieval(self, agent_action):
        # config 5
        if self.__core.decision_mode == "delivery":
            return self.__storage_strategies[0].get_action(self.__core.state)
        else:
            return self.__retrieval_strategies[0].get_action(self.__core.state)

    def __transform_a_selectable_storage_direct_retrieval(self, agent_action):
        # config 6
        if self.__core.decision_mode == "delivery":
            return self.__storage_strategies[agent_action].get_action(self.__core.state)
        else:
            return agent_action

    def __transform_a_selectable_storage_fixed_retrieval(self, agent_action):
        # config 7
        if self.__core.decision_mode == "delivery":
            return self.__storage_strategies[agent_action].get_action(self.__core.state)
        else:
            return self.__retrieval_strategies[0].get_action(self.__core.state)

    def __transform_a_selectable_storage_selectable_retrieval(self, agent_action):
        # config 8
        if self.__core.decision_mode == "delivery":
            return self.__storage_strategies[agent_action].get_action(self.__core.state)
        else:
            return self.__retrieval_strategies[agent_action].get_action(self.__core.state)

    def __transform_a_charging_duration(self, agent_action):
        # config 8
        if self.__core.decision_mode == "charging" or self.__core.decision_mode == "charging_check":
            return agent_action

    def render(self, mode='dummy'):
        raise NotImplementedError

    # <editor-fold desc="Getters">
    @property
    def env_input(self):
        return self.__env_input

    @property
    def core_env(self):
        return self.__core

    @property
    def storage_strategies(self):
        return self.__storage_strategies

    @property
    def retrieval_strategies(self):
        return self.__retrieval_strategies

    @property
    def strategy_configuration(self):
        return self.__strategy_configuration

    def valid_action_mask(self):
        state = self.core_env.state
        agv_id = state.current_agv
        agv = state.agv_manager.agv_index[agv_id]
        battery_level = agv.battery
        # charging_thresholds = np.array(state.params.charging_thresholds)
        # mask = (charging_thresholds == 0) | (charging_thresholds > battery_level)
        mask = [1, 1]
        # If battery level is below 20%, action 0 (Don't charge) is invalid
        if battery_level < 20:
            mask[0] = 0
        return np.array(mask)

    # </editor-fold>

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
