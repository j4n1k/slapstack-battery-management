import math
from collections import deque

import numpy as np
import heapq as heap
from slapstack.core_state import State
from slapstack.core_state_agv_manager import AGV, AgvManager
from slapstack.core_state_location_manager import LocationManager
from slapstack.helpers import faster_deepcopy, StorageKeys, VehicleKeys, \
    BatchKeys, TimeKeys, TravelEventKeys, ravel, unravel
from typing import Tuple, MutableSequence, Set, Dict, Deque, Any, Type, List, \
    Union, TYPE_CHECKING
from slapstack.core_state_route_manager import Route

if TYPE_CHECKING:
    from slapstack.core_state import State


class EventHandleInfo:
    def __init__(
            self,
            action_needed: bool,
            event_to_add,
            travel_event_to_add,
            queued_retrieval_order_to_add,
            queued_delivery_order_to_add,
            queued_charging_event_to_add=None
    ):
        self.action_needed = action_needed
        self.event_to_add = event_to_add
        self.travel_event_to_add = travel_event_to_add
        self.queued_retrieval_order_to_add = queued_retrieval_order_to_add
        self.queued_delivery_order_to_add = queued_delivery_order_to_add
        self.queued_charging_event_to_add = queued_charging_event_to_add


class Event:
    """this class contains all the different types of events in the simulation
    that are the foundation of the processes in the storage location allocation
    problem
    """
    event_counter = 0
    def __init__(self, time: float, verbose: bool):
        """

        time: float
            The time at which travel events should end and the time at which
            orders arrive
        verbose: bool
            debugging, slapstack_controls print statements
        """
        self.time = time
        self.verbose = verbose
        self.id = Event.event_counter
        Event.event_counter += 1

    def __eq__(self, other):
        """used for sorting events in heaps"""
        return self.time == other.time

    def __le__(self, other):
        return self.time <= other.time

    def __lt__(self, other):
        return self.time < other.time

    def __ge__(self, other):
        return self.time >= other.time

    def __gt__(self, other):
        return self.time > other.time

    def handle(self, state: State):
        if self.time > state.time:
            state.time = self.time
        # raise NotImplementedError

    def __hash__(self):
        return hash(str(self.id))


class Order(Event):
    def __init__(self, time: float, SKU: int, order_number: int, batch_id: int,
                 verbose: bool, io_type=None, period=None):
        super().__init__(time, verbose)
        self.SKU = SKU
        self.batch_id = batch_id
        self.order_number = order_number
        self.period = period
        self.completion_time = -1
        self.type = io_type

    def __hash__(self):
        return hash(str(self.order_number))

    def __eq__(self, other):
        """used for sorting events in heaps"""
        if isinstance(other, Order):
            return self.order_number == other.order_number
        else:
            return False

    def handle(self, state: State):
        super().handle(state)
        state.n_skus_inout_now[self.SKU] += 1
        if self.period != state.params.sku_period:
            state.params.sku_period = self.period

    def set_completion_time(self, time: float):
        self.completion_time = time


class Delivery(Order):
    """delivery order contains a SKU number and source position. seizes a free
    AGV, moves it to source to pick up a pallet, then moves it to a desired
    position in the warehouse and drops off the pallet.

    """
    def __init__(self, time: float, SKU: int, order_number: int, verbose: bool,
                 source: int, batch_id: int = 0, period=None,
                 destination: int = None):
        super().__init__(time, SKU, order_number, batch_id, verbose, 'delivery',
                         period)
        self.source = source
        self.destination = destination
        if self.verbose:
            print("event created: ", self)

    def handle(self, state: State, core=None) -> EventHandleInfo:
        """creates a delivery first leg travel event if there are free AGVs,
        updates agv cache. if there are no free AGVs, the event gets added
        to queued_delivery_orders"""
        super().handle(state)
        agv_pos, index = state.agv_manager.get_close_agv(
            state.I_O_positions[self.source], self.type)
        if (state.agv_manager.agv_available('delivery')
                and state.delivery_possible(agv_pos, index)):
            agv: AGV = state.agv_manager.book_agv(
                agv_pos, state.time, index, self.type, core.events)
            if agv.forks > 1 and agv.available_forks < agv.forks - 1:
                existing_event = core.events.find_travel_event(
                    agv.id, DeliveryFirstLeg)
                existing_event.orders.append(self)
                return EventHandleInfo(False, None, None, None, None)
            else:
                state.agv_manager.update_v_matrix(agv.position, None)
                travel_event = DeliveryFirstLeg(
                    state=state, start_point=agv.position,
                    end_point=state.I_O_positions[self.source],
                    travel_type="delivery_first_leg",
                    level=0, source=self.source, orders=[self],
                    order=self, agv_id=agv.id)
                state.add_travel_event(travel_event)
                return EventHandleInfo(
                    False, travel_event, travel_event, None, None)
        else:
            if self.verbose:
                print("no delivery AGV available, adding to queued events")
            return EventHandleInfo(False, None, None, None, self)

    def __str__(self):
        return (f'delivery order #{self.order_number} for SKU {self.SKU} that '
                f'arrives at {self.time} at source {self.source}')

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Retrieval(Order):
    """retrieval order contains a SKU number and sink position. seizes a free
    AGV, moves it to the pallet, picks it up, then moves it to the appropriate
    sink position in the warehouse and drops off the pallet.

    """
    def __init__(self, time: float, SKU: int, order_number: int, verbose: bool,
                 sink: int, batch_id: int = 0, period=None):
        super().__init__(time, SKU, order_number, batch_id, verbose,
                         "retrieval", period)
        self.sink = sink
        if self.verbose:
            print("event created: ", self)

    def handle(self, state: State):
        """3 different scenarios here.
        1) if the SKU number for the retrieval order is not in the warehouse,
        the retrieval order gets added to queued retrieval orders
        2) if the SKU is serviceable and there are free AGVs, action_needed
        returns True so that the agent can create a retrieval order with the
        specific pallet location coming from an agent or retrieval policy
        3) if there are no free AGVs, the retrieval order gets queued and is
        added to the queued_retrieval_orders dictionary
        """
        super().handle(state)
        lm: LocationManager = state.location_manager
        source = None
        # TODO: check and refactor this method!
        if self.SKU not in lm.lane_manager.occupied_lanes:
            if self.verbose:
                print("SKU not available, adding to unserviceable SKUs")
            found_waiting_delivery_order = False
            if self.SKU in lm.events.queued_delivery_orders.keys():
                delivery_order = lm.events.queued_delivery_orders[self.SKU][0]
                if isinstance(delivery_order, Delivery):
                    found_waiting_delivery_order = True
                    source = delivery_order.source
            if not found_waiting_delivery_order:
                return EventHandleInfo(False, None, None, self, None)
            else:
                if state.agv_manager.agv_available():
                    io_locations = State.get_io_locations(state.S)
                    source_location = ravel(
                        io_locations[source] + (0,), state.S.shape)
                    lm.source_location = source_location
                    return EventHandleInfo(True, None, None, None, None)
                else:
                    lm.source_location = None
                    return EventHandleInfo(False, None, None, self, None)
            # state.state_cache.unserviceable_retrieval_skus.add(self.SKU)
            # return EventHandleInfo(False, None, None, self, None)
        if not lm.get_sku_locations(
                self.SKU, state.trackers.travel_event_statistics):
            if self.verbose:
                print("SKU available but is currently locked")
            lm.source_location = None
            return EventHandleInfo(False, None, None, self, None)

        # let step() and agent create the retrieval travel event
        if state.agv_manager.agv_available():
            lm.source_location = None
            return EventHandleInfo(True, None, None, None, None)
        else:
            if self.verbose:
                print("no retrieval AGV available, adding to queued events")
            lm.source_location = None
            return EventHandleInfo(False, None, None, self, None)

    def __str__(self):
        return (f'retrieval order #{self.order_number} for SKU {self.SKU} '
                f'arrives at {self.time} at sink {self.sink}')

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Travel(Event):
    """travel events always have a route and describe what tiles an AGV should
    go through in order to reach a destination
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, orders: [Order], order: Order,
                 key: TravelEventKeys, agv_id: int):
        """
        state: State
        start_point: Tuple[int, int]
            first node/tile in the route
        end_point: Tuple[int, int]
            last node/tile in the route
        SKU: int
            the sku number for the order that the travel event is completing
        travel_type: str
            there are four different types of travel. each is its own object
            but is also described here. values can be: retrieval_first_leg,
            retrieval_second_leg, delivery_first_leg, or delivery_second_leg
        order_number: int
            the order number for the order that the travel event is completing
        verbose: bool
            debugging, slapstack_controls print statements
        level: int
            routes only contain 2D coordinates, so level is saved here.
            for retrieval travel events, defines the level that a pallet is
            being retrieved from. for delivery travel events, defines the
            level where a pallet should be placed.

        """
        self.key = key
        self.agv_id = agv_id
        route = Route(state.routing, start_point, end_point)
        duration = route.get_duration()
        self.matriel_handling_time = state.params.material_handling_time
        super().__init__(state.time + duration + self.matriel_handling_time,
                         order.verbose)
        self.route = route
        state.add_route(route.get_indices())
        self.travel_type = travel_type
        self.level = level
        if self.verbose:
            print("event created: ", self)
            print("route created:")
        # convenience variables set during handle
        self.first_node = start_point
        self.last_node = end_point
        self.order = order
        self.orders = orders
        self.distance_penalty = 0  # from pallet shifts
        state.update_on_travel_event_creation(self.key)
        self.intercepted = False
        self.charging_required = False
        self.charging = False
        self.check_charging = False

    def set_intercept(self, intercepted: bool):
        self.intercepted = intercepted

    def __hash__(self):
        return hash(str(self.order.order_number) + self.travel_type)

    def __eq__(self, other):
        """used for sorting events in heaps"""
        return (self.order.order_number == other.order.order_number
                and self.travel_type == other.travel_type)

    def _check_battery(self, state: 'State'):
        return state.agv_manager.charge_needed(False, self.agv_id)

    def _get_charging_travel(self, state: 'State', core, cs=None):
        agv: AGV = state.agv_manager.agv_index[self.agv_id]
        agv.scheduled_charging = False
        if not cs:
            cs = state.agv_manager.get_charging_station(agv.position, core)
        ChargingFirstLeg.charging_nr += 1
        dummy_charging_order = Order(
            -np.infty, -999, ChargingFirstLeg.charging_nr * -1, -999, False)
        travel_event = ChargingFirstLeg(
            state=state,
            start_point=self.last_node,
            end_point=(cs[0], cs[1]),
            travel_type="charging_first_leg",
            level=0,
            # source=0,
            orders=None,
            order=dummy_charging_order,
            agv_id=self.agv_id,
            core=core)
        state.add_travel_event(travel_event)
        assert travel_event.order.order_number in state.travel_events.keys()
        return travel_event

    def handle(self, state: State, core=None):
        """executed for all types of travel events - correct nodes are set
        based on current route, travel time and distance traveled are calculated
        for trackers"""
        super().handle(state)
        tk = self.key
        d = self.route.get_total_distance()
        t = self.route.get_duration()
        p = self.distance_penalty
        if isinstance(self, DeliverySecondLeg) or isinstance(
                self, RetrievalSecondLeg):
            loaded = True
        else:
            loaded = False
        state.agv_manager.deplete_battery(t, loaded, self.agv_id)
        state.update_on_travel_event_completion(tk, t, d, p)

    def _update_route(self, state: 'State', elapsed_time: float):
        agvm: AgvManager = state.agv_manager
        assert len(agvm.booked_idle_positions) + len(
            agvm.free_idle_positions) == agvm.n_agvs
        if self.travel_type == 'relocation':
            assert self.route.end not in state.agv_manager.free_idle_positions
            assert self.route.end in state.agv_manager.booked_idle_positions
        previous_first_node = self.route.get_first_node()
        state.remove_route(self.route.get_indices())
        self.route.update_path(elapsed_time)
        if self.travel_type == 'relocation':
            assert self.route.end not in state.agv_manager.free_idle_positions
            assert self.route.end in state.agv_manager.booked_idle_positions
        new_first_node = self.route.get_first_node()
        if previous_first_node != new_first_node:
            state.add_route(self.route.get_indices())
        return previous_first_node, new_first_node

    def partial_step_handle(self, state: State, elapsed_time: float):
        # previous_first_node = self.route.get_first_node()
        # state.remove_route(self.route.get_indices())
        # self.route.update_path(elapsed_time)
        # state.add_route(self.route.get_indices())
        # cur_total_time = self.route.get_duration()
        # new_first_node = self.route.get_first_node()
        # state.agv_manager.update_v_matrix(
        #     previous_first_node, new_first_node)
        if len(self.route.get_indices()) <= 1:
            return
        previous_first_node, new_first_node = self._update_route(
            state, elapsed_time)
        if self.travel_type == 'relocation':
            state.agv_manager.update_relocating_agv_position(
                previous_first_node, new_first_node, self.agv_id)
        else:
            state.agv_manager.update_v_matrix(
                previous_first_node, new_first_node)

    def __str__(self):
        return (f'{self.travel_type} travel with SKU {self.order.SKU} finishes '
                f'at {self.time} and takes route {self.route} with duration '
                f'{self.route.get_duration()} to level {self.level}')


class Relocation(Travel):
    relocation_nr = 0

    def __init__(self, state: 'State', start_point: Tuple[int, int],
                 agv_id: int):
        agvm: AgvManager = state.agv_manager
        assert (len(agvm.free_idle_positions) +
                len(agvm.booked_idle_positions) == agvm.n_agvs)
        ids = []
        positions = []
        for pos, agv_l in agvm.free_agv_positions.items():
            positions.append(pos)
            for agv in agv_l:
                ids.append(agv.id)
        assert agv_id in ids
        assert start_point in positions
        end_point = agvm.get_close_idle_position(start_point)
        assert end_point not in agvm.free_idle_positions
        assert end_point in agvm.booked_idle_positions
        Relocation.relocation_nr += 1
        dummy_relocation_order = Order(
            -np.infty, -999, Relocation.relocation_nr, -999, False)
        super().__init__(state, start_point, end_point, 'relocation',
                         0, None, dummy_relocation_order,
                         TravelEventKeys.RELOCATION, agv_id)
        assert self.route.end not in agvm.free_idle_positions
        assert self.route.end in agvm.booked_idle_positions
        agvm.relocating_agvs.add(agv_id)
        self.relocation_update_time = state.time


    def partial_step_handle(self, state: 'State', elapsed_time: float):
        # TODO: can this time be a float? when does it get rounded?
        self.relocation_update_time += elapsed_time
        super().partial_step_handle(state, elapsed_time)

    def handle(self, state: 'State', core=None):
        if self.intercepted:
            return EventHandleInfo(False, None, None, None, None)
        super().handle(state)
        agvm: AgvManager = state.agv_manager
        assert self.agv_id in agvm.relocating_agvs
        # elapsed_time = round(self.time - self.relocation_update_time)
        # previous_first_node, new_first_node = self._update_route(
        #     state, elapsed_time)
        # assert self.route.end not in state.agv_manager.free_idle_positions
        # assert self.route.end in state.agv_manager.occupied_idle_positions
        # agvm.update_relocating_agv_position(
        #     previous_first_node, new_first_node, self.agv_id)
        agvm.update_relocating_agv_position(self.route.get_first_node(),
                                            self.route.get_last_node(),
                                            self.agv_id)
        assert self.last_node == self.route.get_last_node()
        assert self.last_node in agvm.booked_idle_positions
        assert self.last_node not in agvm.free_idle_positions
        # assert self.first_node == self.route.get_first_node()
        agvm.mark_idle_position_arrival(self.last_node, self.agv_id)
        return EventHandleInfo(False, None, None, None, None)

    def __hash__(self):
        return super().__hash__()


class RetrievalFirstLeg(Travel):
    """first leg of a travel event used to complete a retrieval order. a free
     AGV is seized and moves from its current position to the tile from where
    it is picking up a pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, sink: int, orders: [Order],
                 order: Order, actions: List[Tuple[int, int, int]],
                 agv_id: int, servicing_delivery_order: [Delivery]):
        super().__init__(state, start_point, end_point, travel_type,
                         level, orders, order,
                         TravelEventKeys.RETRIEVAL_1STLEG, agv_id)
        self.sink = sink
        self.delivery_orders = servicing_delivery_order
        self.actions = actions
        if len(self.delivery_orders) > 0:
            self.delivery_action = end_point
        else:
            self.delivery_action, self.retrieval_actions = \
                (state.location_manager.lock_lane(end_point))

    def handle(self, state: State, core=None):
        """updates agv position in vehicle matrix, removes sku from storage
        and arrival time matrix, removes SKU from occupied locations in cache,
        removes route, removes travel event, creates retrieval second leg travel
        event.

        returns:
        action_needed = False
        event_to_add = travel_event
        travel_event_to_add = travel_event
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        pallet_position = self.last_node + (self.level,)
        pallet_cycle_time = state.time - state.T[pallet_position]

        shift_penalty = \
            self.update_metrix_lanes(state, pallet_position, pallet_cycle_time)
        assert len(pallet_position) == 3

        state.agv_manager.update_v_matrix(self.first_node, self.last_node)
        state.remove_route(self.route.get_indices())
        state.remove_travel_event(self.order.order_number)
        state.trackers.number_of_pallet_shifts += shift_penalty
        state.n_skus_inout_now[self.order.SKU] -= 1

        index = self.orders.index(self.order)
        if index != len(self.orders) - 1:
            return self.update_current_event_second_order(state, index,
                                                          shift_penalty)
        else:
            travel_event = RetrievalSecondLeg(
                state=state, start_point=self.last_node,
                end_point=state.I_O_positions[self.orders[0].sink],
                travel_type="retrieval_second_leg",
                level=0, sink=self.orders[0].sink,
                n_shifts=shift_penalty, orders=self.orders,
                order=self.orders[0], agv_id=self.agv_id,
                servicing_delivery_orders=self.delivery_orders)
            state.add_travel_event(travel_event)
            return EventHandleInfo(False, travel_event, travel_event,
                                   None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    def update_metrix_lanes(self, state: State,
                            pallet_position: Tuple[int, int],
                            pallet_cycle_time: int):
        shift_penalty = 0
        if hasattr(self, 'retrieval_actions')\
                and self.retrieval_actions is not None:
            state.update_s_t_b_matrices(pallet_position,
                                        StorageKeys.EMPTY.value,
                                        TimeKeys.NAN.value, BatchKeys.NAN.value)
            state.location_manager.unlock_lane(self.delivery_action,
                                               self.retrieval_actions)
            state.location_manager.events.\
                finished_ret_first_leg_skus.append(self.order.SKU)
            shift_penalty = \
                state.location_manager.update_on_retrieval_first_leg(
                    pallet_position, self.order.SKU, pallet_cycle_time,
                    self.time)
        else:
            state.update_s_t_b_matrices(pallet_position,
                                        StorageKeys.SOURCE.value,
                                        TimeKeys.NAN.value, BatchKeys.NAN.value)
        return shift_penalty

    def update_current_event_second_order(self, state: State, index: int,
                                          shift_penalty: int):
        self.order = self.orders[index + 1]
        agv: AGV = state.agv_manager.agv_index.get(self.agv_id)
        if self.order in agv.dcc_retrieval_order:
            action = self.find_closest_sku_loc(state, self.order.SKU)
        else:
            action = self.actions[index + 1]
        if action:
            self.sink = self.order.sink
            super().__init__(state, self.last_node, action[0:2],
                             "retrieval_first_leg", action[2], self.orders,
                             self.order, TravelEventKeys.RETRIEVAL_1STLEG,
                             self.agv_id)
            found = False
            for order in self.delivery_orders:
                if self.order.SKU == order.SKU:
                    found = True
                    self.delivery_action = action[0:2]
                    self.retrieval_actions = None
                    break
            if not found:
                self.delivery_action, self.retrieval_actions =\
                    (state.location_manager.lock_lane(action[0:2]))
            state.add_travel_event(self)
            return EventHandleInfo(False, self, self, None, None)
        else:
            travel_event = RetrievalSecondLeg(
                state=state, start_point=self.last_node,
                end_point=state.I_O_positions[self.orders[index].sink],
                travel_type="retrieval_second_leg",
                level=0, sink=self.orders[index].sink,
                n_shifts=shift_penalty, orders=[self.orders[index]],
                order=self.orders[index], agv_id=self.agv_id,
                servicing_delivery_orders=self.delivery_orders)
            state.add_travel_event(travel_event)
            return EventHandleInfo(False, travel_event, travel_event,
                                   self.order, None)


class RetrievalSecondLeg(Travel):
    """second leg of a travel event used to complete a retrieval order. a busy
     AGV moves from its current position (where it just picked up a pallet) to
     the sink tile where it will drop off the pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, sink: int,
                 n_shifts: int, orders: [Order], order: Order,
                 agv_id: int, servicing_delivery_orders: [Delivery]):
        super().__init__(state, start_point, end_point, travel_type, level,
                         orders, order, TravelEventKeys.RETRIEVAL_2ND_LEG,
                         agv_id)
        self.sink = sink
        self.time += n_shifts * state.params.shift_penalty
        self.distance_penalty = n_shifts * state.params.unit_distance * 2
        self.delivery_orders = servicing_delivery_orders
        state.agv_manager.agv_index[agv_id].servicing_order_type =\
            'retrieval_second_leg'
        self.backtracking_offset = (-1 if self.route.get_indices()[0][0] != 0
                                    else 1)

    def handle(self, state: State, core=None):
        """updates agv position in vehicle matrix, removes route, removes travel
        event.

        Since multiple AGVs can end up at the same spot (sink tile), this
        function tries to make sure that they are placed on different tiles in
        the vehicle matrix.

        The presence of an AGV at the sink tile is marked by the FREE vehicle
        key. Busy AGVs are otherwise allowed to overlap, since they are expected
        to move away in a next step.

        returns:
        action_needed = False
        event_to_add = None
        travel_event_to_add = None
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        # if the AGV has not reached the sink tile yet or if it's already
        # there (possibly from a simulate_travel_events() advancing it
        # to the last node in its route
        tile_found = False
        offset = 0
        while not tile_found:
            vehicle_position = (self.last_node[0], self.last_node[1] + offset)
            value_at_sink = state.V[vehicle_position]
            # if (value_at_sink == VehicleKeys.N_AGV or
            #         value_at_sink == VehicleKeys.BUSY):
            # TODO: implement send back to waiting station scheme ;)
            state.agv_manager.update_v_matrix(self.first_node, self.last_node)
            state.remove_route(self.route.get_indices())
            state.remove_order(self.order.order_number)
            state.remove_travel_event(self.order.order_number)
            state.update_when_vehicle_free(self.last_node)
            tile_found = True
            # else:
            #     if self.verbose:
            #         print("already an AGV at this tile, "
            #               "checking next tile for free space")
            #     offset = self.backtracking_offset
        if self.verbose:
            print(f"finished retrieval order #{self.order.order_number}")

        self.complete_direct_delivery_order(state, core)
        state.location_manager.events.\
            finished_ret_second_leg_skus.append(self.order.SKU)
        self.order.set_completion_time(state.time)
        state.trackers.update_on_order_completion(
            self.order, self.distance_penalty)
        state.agv_manager.agv_index[self.agv_id].orders_since_last_charge += 1
        self.orders.pop(0)

        if len(self.orders) > 0:
            return self.update_current_event_next_order(state)
        else:
            last_node = (self.last_node[0] + offset, self.last_node[1])
            state.agv_manager.update_v_matrix(self.first_node,
                                              self.last_node, True)
            state.agv_manager.agv_index.get(self.agv_id).dcc_retrieval_order\
                = []
            state.agv_manager.release_agv(last_node, state.time, self.agv_id)
            # if super()._check_battery(state):
            #     travel = super()._get_charging_travel(state, core)
            # else:
            #     travel = Relocation(state, self.last_node, self.agv_id)
            # state.add_travel_event(travel)
            # return EventHandleInfo(False, travel, travel, None, None)
            event = GoChargingCheck(self.time, state, self.last_node, self.agv_id, core)
            return EventHandleInfo(False, event, None, None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    def complete_direct_delivery_order(self, state: State, core):
        for order in self.delivery_orders:
            if self.order.SKU == order.SKU:
                delivery_order_index = self.delivery_orders.index(order)
                delivery_order = self.delivery_orders.pop(delivery_order_index)
                delivery_order.set_completion_time(state.time)
                state.trackers.update_on_order_completion(delivery_order)
                state.remove_order(delivery_order.order_number)
                core.logger.log_state()
                state.agv_manager.agv_index[self.agv_id].orders_since_last_charge += 1

    def update_current_event_next_order(self, state: State):
        self.order = self.orders[0]
        self.sink = self.order.sink
        super().__init__(state, self.last_node,
                         state.I_O_positions[self.orders[0].sink],
                         "retrieval_second_leg", 0, self.orders, self.order,
                         TravelEventKeys.RETRIEVAL_2ND_LEG, self.agv_id)
        state.add_travel_event(self)
        return EventHandleInfo(False, self, self, None, None)


class DeliveryFirstLeg(Travel):
    """first leg of a travel event used to complete a delivery order. a free
     AGV is seized and moves from its current position to the source tile where
    it will pick up a pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, source: int, orders: [Order],
                 order: Order, agv_id: int):
        super().__init__(state, start_point, end_point, travel_type,
                         level, orders, order, TravelEventKeys.DELIVERY_1ST_LEG,
                         agv_id)
        self.source = source

    def handle(self, state: State, core=None):
        """vehicle position is updated. since delivery second leg needs an
        action from the agent or policy, it is not created here.

        returns:
        action_needed = True
        event_to_add = None
        travel_event_to_add = None
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        state.remove_travel_event(self.order.order_number)
        state.agv_manager.update_v_matrix(
            self.first_node, self.last_node, False)
        state.remove_route(self.route.get_indices())

        current_order_index = self.orders.index(self.order)
        if current_order_index != len(self.orders) - 1:
            self.order = self.orders[current_order_index + 1]
            super().__init__(state, self.last_node,
                             state.I_O_positions[self.order.source],
                             "delivery_first_leg", 0, self.orders, self.order,
                             TravelEventKeys.DELIVERY_1ST_LEG, self.agv_id)
            self.source = self.order.source
            state.add_travel_event(self)
            return EventHandleInfo(False, self, self, None, None)
        else:
            self.order = self.orders[0]
            return EventHandleInfo(True, None, None, None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class DeliverySecondLeg(Travel):
    """second leg of a travel event used to complete a retrieval order. a busy
     AGV moves from its current position (where it just picked up a pallet) to
     the sink tile where it will drop off the pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, source: int, orders, order, agv_id: int,
                 servicing_retrieval_order: Retrieval = None):
        super().__init__(state, start_point, end_point, travel_type,
                         level, orders, order, TravelEventKeys.DELIVERY_2ND_LEG,
                         agv_id)
        self.source = source
        self.retrieval_order = servicing_retrieval_order
        state.agv_manager.agv_index[agv_id].servicing_order_type = \
            'delivery_second_leg'
        if self.retrieval_order:
            self.delivery_action = end_point
        else:
            self.delivery_action, self.retrieval_actions =\
                (state.location_manager.lock_lane(end_point))

    def handle(self, state: State, core=None):
        """updates agv position in vehicle matrix, removes route, removes travel
        event. adds correct SKU number to storage matrix and the current time
        to the arrival time matrix

        returns:
        action_needed = False
        event_to_add = None
        travel_event_to_add = None
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        self.orders.pop(0)
        state.update_when_vehicle_free(self.last_node)
        state.remove_travel_event(self.order.order_number)
        state.agv_manager.update_v_matrix(self.first_node, self.last_node, True)
        state.remove_route(self.route.get_indices())
        pallet_position = self.last_node + (self.level,)
        if not self.retrieval_order:
            state.location_manager.unlock_lane(self.delivery_action,
                                               self.retrieval_actions)
            state.update_s_t_b_matrices(pallet_position, self.order.SKU,
                                        state.time, self.order.batch_id)
            storage_location = self.last_node + (self.level,)
            state.location_manager.zone_manager.add_out_of_zone_sku(
                pallet_position, self.order.SKU)
            assert len(storage_location) == 3
            # no shift penalty can occur at this point, at least not for the
            # "pure lane" simulation mode
            _ = state.location_manager.update_on_delivery_second_leg(
                pallet_position, self.order.SKU, self.time, self.order.batch_id)
        else:
            self.perform_direct_retrieval(state, core)

        state.remove_order(self.order.order_number)
        state.n_skus_inout_now[self.order.SKU] += 1
        if self.verbose:
            print("finished delivery order #{0}".format(
                self.order.order_number))
        self.order.set_completion_time(state.time)
        state.trackers.update_on_order_completion(self.order)
        state.agv_manager.agv_index[self.agv_id].orders_since_last_charge += 1
        if len(self.orders) > 0:
            self.order = self.orders[0]
            self.source = self.order.source
            return EventHandleInfo(True, None, None, None, None)
        else:
            state.agv_manager.update_v_matrix(
                self.first_node, self.last_node, True)
            agv: AGV = state.agv_manager.agv_index.get(self.agv_id)
            is_exit = False
            for dcc_ret_order in agv.dcc_retrieval_order:
                if self.retrieval_order and dcc_ret_order.SKU ==\
                        self.retrieval_order.SKU:
                    is_exit = True
            if is_exit or len(agv.dcc_retrieval_order) == 0:
                agv.dcc_retrieval_order = []
                state.agv_manager.release_agv(
                    pallet_position[:-1], self.time, self.agv_id)
                # if super()._check_battery(state):
                #     travel = super()._get_charging_travel(state, core)
                # else:
                #     travel = Relocation(state, self.last_node, self.agv_id)
                # state.add_travel_event(travel)
                # return EventHandleInfo(False, travel, travel, None, None)
                event = GoChargingCheck(self.time, state, self.last_node, self.agv_id, core)
                return EventHandleInfo(False, event, None, None, None)
            else:
                actions = []
                pending_ret_orders = []
                possible_ret_orders = []
                for dcc_order in agv.dcc_retrieval_order:
                    action = self.find_closest_sku_loc(state, dcc_order.SKU)
                    if action:
                        possible_ret_orders.append(dcc_order)
                        actions.append(action)
                    else:
                        pending_ret_orders.append(dcc_order)
                if agv.forks > len(actions) > 0 or agv.forks == len(actions):
                    order = possible_ret_orders[0]
                    action = actions[0]
                    travel_event = RetrievalFirstLeg(
                        state=state, start_point=self.last_node,
                        end_point=action[0:2],
                        travel_type="retrieval_first_leg",
                        level=int(action[2]),
                        sink=order.sink,
                        orders=possible_ret_orders,
                        order=order,
                        actions=actions, agv_id=agv.id,
                        servicing_delivery_order=[])
                    state.add_travel_event(travel_event)
                    return EventHandleInfo(False, travel_event, travel_event,
                                           pending_ret_orders, None)
                else:
                    event = agv.dcc_retrieval_order
                    agv.dcc_retrieval_order = []
                    state.agv_manager.release_agv(
                        pallet_position[:-1], self.time, self.agv_id)
                    travel = Relocation(state, self.last_node, self.agv_id)
                    return EventHandleInfo(False, travel, travel, event, None)

    def find_closest_sku_loc(self, state: State, sku: int)\
            -> Tuple[int, int, int]:
        sc, tes = state.location_manager, state.trackers.travel_event_statistics
        locations = sc.get_sku_locations(sku, tes)
        closest_distance = np.infty
        closest_tgt_loc = None
        for loc in list(locations):
            loc_tuple = unravel(loc, state.S.shape)
            distance = state.routing.get_distance(self.last_node, loc_tuple[:2])
            if distance < closest_distance:
                closest_distance = distance
                closest_tgt_loc = loc_tuple
        return closest_tgt_loc

    def get_next_sku(self):
        return self.order.sku

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    def perform_direct_retrieval(self, state: State, core):
        tile_found = False
        offset = 0
        state.remove_route(self.route.get_indices())
        while not tile_found:
            vehicle_position = (self.last_node[0] + offset, self.last_node[1])
            value_at_sink = state.V[vehicle_position]
            if (value_at_sink == VehicleKeys.N_AGV or
                    value_at_sink == VehicleKeys.BUSY):
                state.agv_manager.update_v_matrix(
                    self.first_node, self.last_node, False)
                state.remove_route(self.route.get_indices())
                state.remove_order(self.retrieval_order.order_number)
                tile_found = True
            else:
                if self.verbose:
                    print("already an AGV at this tile, "
                          "checking next tile for free space")
                offset += 1
        if self.verbose:
            print(f"finished retrieval order #{self.retrieval_order.order_number}")
        self.retrieval_order.set_completion_time(state.time)
        state.trackers.update_on_order_completion(self.retrieval_order)
        state.agv_manager.agv_index[self.agv_id].orders_since_last_charge += 1
        core.logger.log_state()


class ChargingFirstLeg(Travel):
    charging_nr = 0

    def __init__(self, state: 'State', start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, orders, order, agv_id: int, core,
                 servicing_retrieval_order: Retrieval = None, fixed_charging_duration=None,
                 charging_check_time=None):
        super().__init__(state, start_point, end_point, travel_type,
                         level, orders, order,
                         TravelEventKeys.CHARGING_FIRST_LEG, agv_id)
        self.fixed_charging_duration = fixed_charging_duration
        self.charging_check_time = charging_check_time
        if state.agv_manager.agv_index[agv_id].free:
            locs = state.agv_manager.get_agv_locations()
            # assert start_point in locs.keys()
            idx = 0
            target_idx = None
            for agv in locs[start_point]:
                if agv.id == agv_id:
                    target_idx = idx
                else:
                    idx += 1
            # end_pos = state.agv_manager.get_close_idle_position(start_point)
            self.agv: AGV = state.agv_manager.book_agv(
                start_point, state.time, target_idx,
                self.travel_type, core.events, charging_booking=True)
        else:
            self.agv = state.agv_manager.agv_index[agv_id]
        # self.agv.charging_needed = True
        # state.agv_manager.n_charging_agvs += 1
        state.agv_manager.log_travel_to_cs(self.last_node, self.agv)
        assert self.agv.battery > 10
        assert self.agv.free == False
        assert self.agv.id == agv_id

    def handle(self, state: 'State', core=None):
        super().handle(state)
        self.charging = True
        try:
            state.remove_travel_event(self.order.order_number)
        except KeyError:
            print("no travel")
        state.agv_manager.release_travel_to_cs(self.last_node, self.agv)
        state.agv_manager.update_v_matrix(self.first_node, self.last_node, True)
        state.remove_route(self.route.get_indices())
        assert self.last_node in np.argwhere(state.agv_manager.router.s[
                                             :, :, 0] ==
                                             StorageKeys.CHARGING_STATION)
        # if cs not booked -> book and create charging event
        state.agv_manager.book_charging_station(self.last_node, self.agv)
        assert self.agv in state.agv_manager.booked_charging_stations[
            self.last_node]
        assert self.charging == True
        return EventHandleInfo(True, None, None, None, None)


class Charging(Event):
    def __init__(self, time: int, charging_event_travel: ChargingFirstLeg,
                 charging_duration: int, charging_check_time=None):
        self.start_time = time
        self.charging_check_time = charging_check_time
        end_time = time + charging_duration
        super().__init__(time=end_time, verbose=False)
        self.charging_event_travel = charging_event_travel
        self.agv_id = charging_event_travel.agv_id
        self.cs_pos = charging_event_travel.last_node
        self.charging_duration = charging_duration
        self.intercepted = False
        # TODO ensure AMRs don't overlap at charging station

    def check_time_elapsed(self, state: 'State'):
        return state.time - self.start_time

    def check_battery_charge(self, state: 'State'):
        agvm = state.agv_manager
        agv = agvm.agv_index[self.agv_id]
        elapsed_time = self.check_time_elapsed(state)
        battery_increase = 0
        if elapsed_time > 0:
            battery_increase = elapsed_time * agvm.charging_rate
        return agv.battery + battery_increase

    def forced_handle(self, state: 'State', target_battery: int):
#        assert state.time == self.time
        agvm = state.agv_manager
        agv = agvm.agv_index[self.agv_id]
        agv.n_charging_stops += 1
        agvm.release_agv(
            self.cs_pos, state.time, self.agv_id)
        try:
            assert agv.battery <= target_battery
        except:
            print("soc less then target")
        agv.battery = target_battery
        agv.charging_needed = False
        agvm.n_charging_agvs -= 1
        agvm.release_charging_station(self.cs_pos, agvm.agv_index[self.agv_id])
        travel = Relocation(state, self.cs_pos, self.agv_id)
        # charging_event = None
        # queued_event = agvm.queued_charging_events[self.cs_pos][0]
        # try:
        #     assert self == queued_event
        # except:
        #     print()
        # #if agvm.queued_charging_events[self.cs_pos]:
        # _ = agvm.queued_charging_events[self.cs_pos].popleft()
        # if agvm.queued_charging_events[self.cs_pos]:
        #     next_charging = agvm.queued_charging_events[self.cs_pos][0]
        #     try:
        #         assert next_charging.start_time == self.time
        #     except:
        #         print()

        # remove current charging event
        _ = agvm.queued_charging_events[self.cs_pos].popleft()
        assert _ == self
        next_charging_event = None
        if agvm.queued_charging_events[self.cs_pos]:
            # if queued charging events: pull first and return -> to be added to heap
            next_charging_event = agvm.queued_charging_events[self.cs_pos][0]  # .popleft()
            assert (next_charging_event.agv_id ==
                    agvm.booked_charging_stations[self.cs_pos][0].id)
            if next_charging_event.start_time > state.time:
                next_charging_event.start_time = state.time
                next_charging_event.time = state.time + next_charging_event.charging_duration
            assert next_charging_event.start_time == state.time

        # else:
        #     for cs in agvm.queued_charging_events.keys():
        #         if agvm.queued_charging_events[cs]:
        #             charging_event = agvm.queued_charging_events[cs].popleft()
        #             break
        agvm.update_trackers_on_charging_end()
        return travel, next_charging_event

    def handle(self, state: 'State', core=None):
        if not self.intercepted:
            super().handle(state)
            agv = state.agv_manager.agv_index[self.agv_id]
            target_battery = (self.charging_duration * state.agv_manager.charging_rate) + agv.battery
            travel, charging_event = self.forced_handle(state, target_battery)
            return EventHandleInfo(False, travel, travel, None,
                                   None, charging_event)
        else:
            return EventHandleInfo(False, None, None, None, None)
        # agv = state.agv_manager.agv_index[self.agv_id]
        # agv.n_charging_stops += 1
        # # assert agv.charging_needed
        # # TODO check KPI evt. Fallunterschiedung KPI charging nicht travel release
        #
        # state.agv_manager.release_agv(
        #     self.cs_pos, self.time, self.agv_id)
        # # state.agv_manager.charge_battery(self.charging_duration,
        # #                                  agv_id=self.agv_id)
        # state.agv_manager.release_charging_station(self.cs_pos,
        #                                            state.agv_manager.agv_index[
        #                                                self.agv_id])
        # travel = Relocation(state, self.cs_pos, self.agv_id)
        # charging_event = None
        # if state.agv_manager.queued_charging_events[self.cs_pos]:
        #     charging_event = state.agv_manager.queued_charging_events[
        #         self.cs_pos].pop()
        # else:
        #     for cs in state.agv_manager.queued_charging_events.keys():
        #         if state.agv_manager.queued_charging_events[cs]:
        #             charging_event = state.agv_manager.queued_charging_events[
        #                 cs].pop()
        #             break
        # state.agv_manager.update_trackers_on_charging_end()
        #assert charging_event


class GoChargingCheck(Event):
    def __init__(self, time: float, state: 'State', start_point: Tuple[int, int],
                 agv_id: int, core):
        super().__init__(time=time, verbose=False)
        self.agv_id = agv_id
        locs = state.agv_manager.get_agv_locations()
        idx = 0
        target_idx = None
        for agv in locs[start_point]:
            if agv.id == agv_id:
                target_idx = idx
            else:
                idx += 1
        self.agv: AGV = state.agv_manager.book_agv(
            start_point, state.time, target_idx,
            "charging_check", core.events)

        assert self.agv.free == False
        assert self.agv.id == agv_id
        self.last_node = start_point

    def handle(self, state: 'State', core=None):
        super().handle(state)
        return EventHandleInfo(True, None, None, None,
                               None, None)


class EventManager:
    def __init__(self):
        self.running: MutableSequence[Event] = []
        self.queued_delivery_orders: Dict[int, Deque[Delivery]] = {}
        self.queued_retrieval_orders: Dict[int, Deque[Retrieval]] = {}
        self.current_travel: Set[Travel] = set({})
        self.__verbose = False
        self.__retrieval_possible = False
        self.__delivery_possible = False
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        self.__n_queued_retrieval_orders = 0
        self.__n_queued_delivery_orders = 0
        self.__earliest_delivery_order = None
        self.__earliest_retrieval_order = None
        self.finished_ret_first_leg_skus = []
        self.finished_ret_second_leg_skus = []

    @property
    def n_queued_retrieval_orders(self):
        return self.__n_queued_retrieval_orders

    @property
    def n_queued_delivery_orders(self):
        return self.__n_queued_delivery_orders

    @n_queued_retrieval_orders.setter
    def n_queued_retrieval_orders(self, n_orders):
        self.__n_queued_retrieval_orders = n_orders

    @n_queued_delivery_orders.setter
    def n_queued_delivery_orders(self, n_orders):
        self.__n_queued_delivery_orders = n_orders

    def add_future_event(self, event: Event):
        if event is None:
            return
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        heap.heappush(self.running, event)

    def add_current_events(
            self, event_queueing_info: EventHandleInfo, state):
        """this function takes the events that were returned from handling an
        event and adds them to their appropriate data structures. For example,
        if a travel event was created when handling a delivery order, it will
        be added to self.events.current_travel.
        """
        if event_queueing_info.event_to_add:
            self.add_future_event(event_queueing_info.event_to_add)
        if event_queueing_info.travel_event_to_add:
            self.__state_changed_retrieval = True
            self.__state_changed_delivery = True
            initial_len = len(self.current_travel)
            self.current_travel.add(event_queueing_info.travel_event_to_add)
            assert initial_len + 1 == len(self.current_travel)
        if event_queueing_info.queued_retrieval_order_to_add:
            self.__print(f"added retrieval order to queue: "
                         f"{event_queueing_info.queued_retrieval_order_to_add}")
            if type(event_queueing_info.queued_retrieval_order_to_add) == list:
                for queue_ret_order_to_add in\
                        event_queueing_info.queued_retrieval_order_to_add:
                    self.__queue_retrieval_order(queue_ret_order_to_add)
            else:
                self.__queue_retrieval_order(
                    event_queueing_info.queued_retrieval_order_to_add)
        if event_queueing_info.queued_delivery_order_to_add:
            self.__print(f"added delivery order to queue: "
                         f"{event_queueing_info.queued_delivery_order_to_add}")
            self.__queue_delivery_order(
                event_queueing_info.queued_delivery_order_to_add)
        if event_queueing_info.queued_charging_event_to_add:
            self.__print(f"added charging event to queue: "
                         f"{event_queueing_info.queued_charging_event_to_add}")
            self.push_charging_event(
                event_queueing_info.queued_charging_event_to_add)
            assert len(self.running) > 0

    def __queue_delivery_order(self, order: Delivery):
        self.__state_changed_delivery = True
        self.__n_queued_delivery_orders += 1
        sku = order.SKU
        if sku in self.queued_delivery_orders:
            self.queued_delivery_orders[sku].append(order)
        else:
            self.queued_delivery_orders[sku] = deque([order])
        # self.queued_delivery_orders.append(order)

    def __queue_retrieval_order(self, order: Retrieval):
        self.__n_queued_retrieval_orders += 1
        self.__state_changed_retrieval = True
        sku = order.SKU
        if sku in self.queued_retrieval_orders:
            self.queued_retrieval_orders[sku].append(order)
        else:
            self.queued_retrieval_orders[sku] = deque([order])

    def push_charging_event(self, event: Charging):
        heap.heappush(self.running, event)

    def pop_future_event(self):
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        return heap.heappop(self.running)

    def pop_retrieval_event(self, event: RetrievalFirstLeg):
        heap.heapify(self.running)
        return heap.heappop(self.running, event)

    def re_heapify_heap(self):
        heap.heapify(self.running)

    def pop_queued_event(self, state: State) -> Event:
        """this method is only executed if there are queued orders that can be
        handled and there are free AGVs. It is a bit hairy because of many
        specific if statements but these are the three conditions below. Note
        that retrieval orders have higher priority than delivery orders.
        1) if there is at least one serviceable retrieval order, handle
        whichever order one is oldest (i.e. queued first)
        2) if there are no queued retrieval orders, but there are queued
        delivery orders that can be serviced (i.e. there is space in the
        warehouse/ there are legal actions), handle the oldest delivery order
        3) if there are both serviceable queued retrieval orders and queued
        delivery orders, handle whichever one is oldest.
        Finally, since the queued events time are in the past, they are updated
        to be the current state time.
        return the event to be handled
        """
        next_event = None
        if self.__state_changed_retrieval:
            self.available_retrieval(state)
        if self.__state_changed_delivery:
            self.available_delivery(state)
        self.__print("picking from queued events")
        if self.__retrieval_possible and not self.__delivery_possible:
            next_event = self.pop_queued_retrieval_order()
        elif self.__delivery_possible and not self.__retrieval_possible:
            next_event = self.pop_queued_delivery_order()
        elif self.__retrieval_possible and self.__delivery_possible:
            if (self.__earliest_retrieval_order.time
                    <= self.__earliest_delivery_order.time):
                next_event = self.pop_queued_retrieval_order()
            else:
                next_event = self.pop_queued_delivery_order()
        # the queued order's time has already passed so it must be
        # updated to current time
        # TODO: Fix service times?
        # if next_event:
        #     next_event.time = state.time
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        return next_event

    def add_travel_event(self, event: Travel):
        """
        Adds the travel event to both the future events queue and the running
        travel events queue.

        :param event: The new travel event wich is to finish at some time in the
            future.
        :return: None.
        """
        self.current_travel.add(event)
        self.add_future_event(event)

    def find_travel_event(
            self, agv_id: int, event_type: Type[Travel]
    ) -> Union[DeliveryFirstLeg, DeliverySecondLeg,
               RetrievalFirstLeg, RetrievalSecondLeg, Relocation, Travel]:
        """
        Finds an already running travel event given the agv_id and the type of
        the desired event.

        When multiple forks are involved, already running RetrievalFirstLeg
        events sometimes need to be updated to service additional orders.

        :param agv_id: The target agv.
        :param event_type: The type of travel event.
        :return: The desired Travel event.
        """
        if type(event_type) == str:
            event_type = Relocation
        for event in self.current_travel:
            if isinstance(event, event_type) and event.agv_id == agv_id:
                return event

    def remove_travel_event(self, next_event: Travel):
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        initial_len = len(self.current_travel)
        self.current_travel.remove(next_event)
        assert initial_len - 1 == len(self.current_travel)

    def available_retrieval(self, state: State):
        if not self.__state_changed_retrieval:
            return self.__retrieval_possible
        query_result = False
        if self.__n_queued_retrieval_orders > 0:   # there are queued orders
            minimum_time = math.inf
            oldest_order_possible = None
            for sku, retrieval_orders in self.queued_retrieval_orders.items():
                first_order_time = retrieval_orders[0].time
                if state.retrieval_possible(sku):
                    if first_order_time < minimum_time:
                        minimum_time = first_order_time
                        oldest_order_possible = retrieval_orders[0]
            if oldest_order_possible:
                self.__earliest_retrieval_order = oldest_order_possible
                query_result = True
        self.__state_changed_retrieval = False
        self.__retrieval_possible = query_result
        return query_result

    def get_retrieval_possible(self):
        return self.__retrieval_possible

    def available_delivery(self, state: State):
        """

        :param state:
        :return:
        """
        if not self.__state_changed_delivery:
            return self.__delivery_possible
        if len(self.queued_delivery_orders) == 0:
            query_result = False
        else:
            query_result = state.delivery_possible()
            sku = next(iter(self.queued_delivery_orders))
            self.__earliest_delivery_order = self.queued_delivery_orders[sku][0]
        self.__delivery_possible = query_result
        self.__state_changed_delivery = False
        return query_result

    def all_orders_complete(self):
        if (not self.running
                and self.__n_queued_retrieval_orders == 0
                and self.__n_queued_delivery_orders == 0):
            return True
        else:
            return False

    def pop_queued_delivery_order(self) -> Delivery:
        """
        Returns the oldest serviceable queued delivery order.

        :return: The oldest order.
        """
        self.__state_changed_retrieval = True
        self.__n_queued_delivery_orders -= 1
        oldest_sku = next(iter(self.queued_delivery_orders))
        order = self.queued_delivery_orders[oldest_sku].popleft()
        if len(self.queued_delivery_orders[oldest_sku]) == 0:
            del self.queued_delivery_orders[oldest_sku]
        return order

    def pop_queued_retrieval_order(self) -> Retrieval:
        """
        Return the oldest serviceable retrieval order and removes it from the
        queue. Since the queued retrieval orders are stored in a sku indexed
        dictionary of deques, and orders are queued by appending to the right,
        we only need to look at all the deque heads to find the oldest queued
        order.

        :return: The oldest serviceable retrieval order.
        """
        assert not self.__state_changed_retrieval
        self.__state_changed_delivery = True
        self.__n_queued_retrieval_orders -= 1
        target_sku = self.__earliest_retrieval_order.SKU
        order = self.queued_retrieval_orders[target_sku].popleft()
        if not self.queued_retrieval_orders[target_sku]:
            del self.queued_retrieval_orders[target_sku]
        return order

    def get_earliest_retrieval_order(self) -> Retrieval:
        target_sku = self.__earliest_retrieval_order.SKU
        order = self.queued_retrieval_orders[target_sku][0]
        return order

    def get_min_retrival_order_time(self):
        """
        Computes the minimum retrieval order time.

        :return: The time of the oldest retrieval order.
        """
        min_time = np.inf
        for orders in self.queued_retrieval_orders.values():
            if orders[0].time < min_time:
                min_time = orders[0].time
        return min_time

    def __print(self, string: Any):
        """this function can be used instead of the python default print(). It
        allows all print statements to be turned on/off with one parameter:
        __verbose
        """
        if self.__verbose:
            print(string)

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
