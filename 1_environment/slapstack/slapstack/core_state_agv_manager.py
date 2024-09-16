from collections import defaultdict
from math import inf, hypot
import typing
from typing import Tuple, Dict, List, Union, Set, Deque

import numpy as np

from slapstack.core_state_route_manager import RouteManager

from slapstack.helpers import VehicleKeys, StorageKeys
from slapstack.interface_templates import SimulationParameters
if typing.TYPE_CHECKING:
    from slapstack.core_events import Charging, EventManager


class AGV:
    def __init__(self, transport_id: int, pos: Tuple[int, int], forks=1):
        """
        AGV object constructor. Instances of this class represent the warehouse
        transports.

        :param transport_id: The unique transport id.
        :param pos: The current AGV position.
        """
        self.scheduled_charging = False
        self.id = transport_id
        self.position = pos
        self.free = True
        self.booking_time = -1
        self.utilization = 0
        self.forks = forks
        self.servicing_order_type = None
        self.available_forks = self.forks
        self.dcc_retrieval_order = []
        self.battery = 100
        self.charging_needed = False
        self.n_charging_stops = 0

    def log_booking(self, booking_time: float):
        """
        Marks the AGV as busy and notes down the time at which the booking
        occurred.

        :param booking_time: The time at which the AGV was selected for a job.
        :return: None.
        """
        self.booking_time = booking_time
        self.free = False

    def log_release(self, release_time: float, position: Tuple[int, int]):
        """
        Marks the AGV as free and computes the utilization time using the
        previously set booking time and updates its position.

        :param release_time: The time at which the AGV finished its job.
        :param position: The new AGV position.
        :return: None.
        """
        self.position = position
        self.free = True
        self.utilization += release_time - self.booking_time


class AgvManager:
    def __init__(
            self, p: SimulationParameters, storage_matrix: np.ndarray, rng,
            router: RouteManager):
        """
        AgvManager Constructor. This object deals with information related to
        AGVs through the warehouse. Most importantly it tracks the position
        of the free AGVs and the AGV utilization time.

        The free_agv_positions property indexes free ASVs (@see the AGV object)
        by their position in the warehouse. The agv_index property indexes the
        *same* AGV objects by their id.

        Additionally the class maintains several counters such as the current
        number of free or busy AGVs.

        :param storage_matrix: The vehicle matrix from which to extract the
            initial AGV positions.
        """
        self.consumption_rate_unloaded = p.consumption_rate_unloaded
        self.consumption_rate_loaded = p.consumption_rate_loaded
        self.charging_rate = p.charging_rate
        self.charge_in_break_started = False
        self.free_agv_positions: Dict[Tuple[int, int], List[AGV]] = {}
        self.agv_index: Dict[int, AGV] = {}
        self.V = np.full((p.n_rows, p.n_columns),
                         VehicleKeys.N_AGV.value, dtype='int8')
        self.maximum_forks = p.n_forks
        self.free_idle_positions: Set[Tuple[int, int]] = set({})
        self.booked_idle_positions: Set[Tuple[int, int]] = set({})
        self.stationary_amr_positions: Set[Tuple[int, int]] = set()
        self.relocating_agvs: Set[int] = set({})
        self.__initialize_agvs(storage_matrix, rng, p.n_agvs)
        self.n_free_agvs = len(self.free_agv_positions)
        self.n_busy_agvs = 0
        self.n_agvs = p.n_agvs
        self.n_visible_agvs = p.n_agvs
        self.booked_charging_stations: Dict[Tuple[int, int], List[AGV]] = {}
        self.charging_stations: List[Tuple[int, int]] = list(tuple(map(
            tuple, np.argwhere(storage_matrix[:, :, 0]
                               == StorageKeys.CHARGING_STATION))))
        self.free_charging_stations: Set[Tuple[int, int]] = set(tuple(map(
            tuple, np.argwhere(storage_matrix[:, :, 0]
                               == StorageKeys.CHARGING_STATION))))
        self.queued_charging_events: Dict[Tuple[int, int], Deque[Charging]] = {
            cs: [] for cs in self.free_charging_stations}
        self.router = router
        self.n_charging_stations = len(np.argwhere(self.router.s[:, :, 0]
                                                   == StorageKeys.CHARGING_STATION))
        print(self.n_charging_stations)
        self.agv_trackers = AGVTrackers(self)


    @staticmethod
    def __get_n_agv_xy(n_agvs, arr):
        if n_agvs == 1:
            return 1, 1
        r = (arr.shape[0] / arr.shape[1])
        n_y = np.floor(np.sqrt(n_agvs / r))
        n_x = n_agvs // n_y
        n_x = n_x + 1 if n_x * n_y != n_agvs else n_x
        return int(n_x), int(n_y)

    def __initialize_agvs(self, S: np.ndarray, rng, n_agvs):
        """
        Initializes the simulation AGVs by assigning initial positions (aisle
        only!) and populating the agv_index and free_agv_positions fields.

        :return: None.
        """
        # agv_counter = 2  # initial position (column) of AGVs
        # storage_locations = np.argwhere(S[:, :, 0] == StorageKeys.MID_AISLE)
        # transport_id = 0
        # for i in range(0, n_agvs):
        #     index = rng.integers(0, len(storage_locations))
        #     pos_t = tuple(storage_locations[index])
        #     no_forks = self.maximum_forks
        #     if pos_t in self.free_agv_positions:
        #         continue
        #     new_agv = AGV(transport_id, pos_t, no_forks)
        #     self.free_agv_positions[pos_t] = [new_agv]
        #     self.agv_index[transport_id] = new_agv
        #     transport_id += 1
        #     self.V[pos_t] = VehicleKeys.FREE.value
        aisle_locations = np.argwhere(S[:, :, 0] == StorageKeys.MID_AISLE)
        n_x, n_y = AgvManager.__get_n_agv_xy(n_agvs, S)
        xs_idle_agv, ys_idle_agv = AgvManager.__get_positions(
            aisle_locations, n_x, n_y, n_agvs)
        transport_id = 0
        for i in range(0, n_agvs):
            pos_t = (xs_idle_agv[i], ys_idle_agv[i])
            no_forks = self.maximum_forks
            new_agv = AGV(transport_id, pos_t, no_forks)
            self.booked_idle_positions.add(pos_t)
            self.stationary_amr_positions.add(pos_t)
            self.free_agv_positions[pos_t] = [new_agv]
            self.agv_index[transport_id] = new_agv
            transport_id += 1
            self.V[pos_t] = VehicleKeys.FREE.value

    @staticmethod
    def __find_equal_segment_split_pts(n_pts, n_segments):
        if n_segments == 1:
            return [int(np.floor(n_pts / 2))]
        if n_segments % 2 == 0:
            # assert 2 * n_segments + 1 <= n_pts
            split_pts = np.around(
                np.linspace(0, n_pts - 1, 2 * n_segments + 1)).astype(
                int).flatten()
            return split_pts[1::2].tolist()
        else:
            split_pts = np.around(
                np.linspace(0, n_pts - 1, n_segments)).astype(int).flatten()
            return split_pts.tolist()

    @staticmethod
    def __get_aisle_rows(aisle_tiles):
        """
        Creates dictionary mapping aisle row coordinates to aisle column
        coordinates.

        :param aisle_tiles: The tiles to extract coordinates from.
        :return:
        """
        row_keys = []
        rows = defaultdict(list)
        row_lens = defaultdict(int)
        for tile in aisle_tiles:
            if tile[0] not in rows:
                row_keys.append(tile[0])
            rows[tile[0]].append(tile[1])
            row_lens[tile[0]] += 1
        return row_keys, row_lens, rows

    @staticmethod
    def __filter_long_aisle_rows(rows, row_keys, row_lens):
        lens = np.array(list(row_lens.values()))
        remaining_keys = []
        long_aisle_len = lens.max(initial=0) * 0.75
        for x in row_keys:
            if row_lens[x] < long_aisle_len:
                del rows[x]
                del row_lens[x]
            else:
                remaining_keys.append(x)
        return remaining_keys

    @staticmethod
    def __get_positions(aisle_tiles, n_x, n_y, n_agvs):
        """
        Calculates the positions AGVs return to when idle. The function
        distributes AGVs approximately uniformly within the warehouse.

        :param aisle_tiles: The tiles AGVs can travel over.
        :param n_x: The number of rows to define idle positions on.
        :param n_y: The number of columns to define idle positions on.
        :param n_agvs: The total number of idle positions.
        :return: Corresponding lists of x and y indices of the AMR idle
            positions.
        """
        row_keys, row_lens, rows = AgvManager.__get_aisle_rows(aisle_tiles)
        # the following function modifies the row dictionary!
        long_row_keys = AgvManager.__filter_long_aisle_rows(
            rows, row_keys, row_lens)

        if len(long_row_keys) < n_x:
            n_x = len(long_row_keys)
            n_y = n_agvs // n_x
            if n_agvs % n_x != 0:
                n_y += 1

        # distribute column positions equidistantly for every row
        selected_row_idxs = AgvManager.__find_equal_segment_split_pts(
            len(long_row_keys), n_x)
        xs = []
        ys = []
        for idx_x in selected_row_idxs[:]:
            x = long_row_keys[idx_x]
            selected_col_idxs = AgvManager.__find_equal_segment_split_pts(
                len(rows[x]), n_y)
            for idx_y in selected_col_idxs[:]:
                if len(xs) == n_agvs:
                    break
                xs.append(x)
                ys.append(rows[x][idx_y])
        return xs, ys

    def update_v_matrix(self, first_position: Tuple[int, int],
                        second_position: Union[Tuple[int, int], None],
                        release: bool = False):
        """
        Updates the vehicle matrix as AGVs move across tiles. The following
        cases are distinguished:
        1. When an AGV gets selected to service an order (delivery or retrieval)
        it will be marked as busy. The service booking is indicated by the None
        value of the second_position parameter.
        2. When an AGV simply moves without having finished its order
        (release == False), its position is updated without changing the
        markings status (AGV stays busy).
        3. Whenever an order is finished, the AGVs position is updated (at sink
        for delivery orders, in the corresponding lane for retrieval orders) and
        the vehicle is marked as free.

        :param first_position: The AGV position before the update.
        :param second_position: The AGV position after the update; if None, then
            the AGV has not moved.
        :param release: Whether the AGV became free or not.
        :return: None.
        """
        # when creating an order
        src_x, src_y = first_position[0], first_position[1]
        if not second_position:
            self.V[src_x, src_y] = VehicleKeys.BUSY.value
        else:
            tgt_x, tgt_y = second_position[0], second_position[1]
            if release:
                self.V[src_x, src_y] = VehicleKeys.N_AGV.value
                self.V[tgt_x, tgt_y] = VehicleKeys.FREE.value
            else:
                self.V[src_x, src_y] = VehicleKeys.N_AGV.value
                self.V[tgt_x, tgt_y] = VehicleKeys.BUSY.value

    def update_on_retrieval_second_leg(
            self, position: Tuple[int, int], system_time: float, agv_id: int):
        self.release_agv(position, system_time, agv_id)

    def agv_available(self, order_type='retrieval', only_new=False) -> bool:
        """
        Checks if there are any available AGVs to use for transport.

        :return: True if the free_agv_positions dictionary is not empty and
            false otherwise.
        """
        is_free_agv = False
        for agv_pos in self.free_agv_positions:
            for current_agv in self.free_agv_positions[agv_pos]:
                if only_new:
                    if current_agv.servicing_order_type is None:
                        is_free_agv = True
                elif current_agv.servicing_order_type == order_type\
                        or current_agv.servicing_order_type is None:
                    is_free_agv = True
                    break
            if is_free_agv:
                break
        return is_free_agv

    def update_relocating_agv_position(self, src_pos, dest_pos, agv_id):
        src_x, src_y = src_pos[0], src_pos[1]
        tgt_x, tgt_y = dest_pos[0], dest_pos[1]
        # update v matrix
        self.V[tgt_x, tgt_y] = self.V[src_x, src_y]
        self.V[src_x, src_y] = VehicleKeys.N_AGV.value
        # update free agv index
        if len(self.free_agv_positions[src_pos]) > 1:
            for agv in self.free_agv_positions[src_pos]:
                if agv.id == agv_id:
                    self.free_agv_positions[src_pos].remove(agv)
                    break
        else:
            fagv = self.free_agv_positions[src_pos].pop()
            assert fagv.id == agv_id
        if not self.free_agv_positions[src_pos]:
            del self.free_agv_positions[src_pos]

        relocated_agv = self.agv_index[agv_id]
        relocated_agv.position = dest_pos
        if dest_pos not in self.free_agv_positions:
            self.free_agv_positions[dest_pos] = [relocated_agv]
        else:
            self.free_agv_positions[dest_pos].append(relocated_agv)

    def book_agv(self, agv_pos: Tuple[int, int], system_time: float,
                 free_agv_index: int, order_type: str,
                 event_manager: 'EventManager'
                 ):
        """
        Called whenever a fist leg transport event starts. Depending on the
        event trigger position (source for delivery first leg or chosen pallet
        in the case of retrieval first leg) the closest agv from the
        free_agv_positions is selected for booking. The euclidean distance is
        used for distance comparison.

        An AGV object located at the chosen position is removed from
        free_agv_positions and its booking time is marked by calling the
        log_booking method. Finally, if the chosen AGV was the last located at
        the given position, the position is removed from the index entirely.

        :param position: The position of the triggering event (either source or
            chosen sku).
        :param system_time: The current simulation time.
        #TODO reword docstring comments to reflect the external agv selection
        :return: The AGV booked AGV.
        """
        assert (len(self.free_idle_positions) +
                len(self.booked_idle_positions) == self.n_agvs)
        agv = self.free_agv_positions[agv_pos][free_agv_index]
        if agv.available_forks == agv.forks:
            agv.log_booking(system_time)
            agv.servicing_order_type = order_type
        agv.available_forks -= 1
        if agv.available_forks == 0:
            self.n_busy_agvs += 1
            self.n_free_agvs -= 1
            self.free_agv_positions[agv_pos].pop(free_agv_index)
        if not self.free_agv_positions[agv_pos]:
            del self.free_agv_positions[agv_pos]
        if agv.id in self.relocating_agvs:
            print()
            travel_e = event_manager.find_travel_event(agv.id, 'relocation')
            assert not travel_e.intercepted
            travel_e.set_intercept(True)
            assert travel_e.last_node not in self.free_idle_positions
            assert travel_e.last_node in self.booked_idle_positions
            event_manager.remove_travel_event(travel_e)
            # remove the Relocation Event from the heap!!!!
            self.relocating_agvs.remove(agv.id)
            self.free_idle_positions.add(travel_e.last_node)
            self.booked_idle_positions.remove(travel_e.last_node)
        else:
            # TODO differentiate between charging and relocation booking
            #  and reintegrate assertion
            # try:
            #     # assert agv_pos in self.booked_idle_positions
            # except AssertionError:
            #     pass
            # if (not agv.servicing_order_type == "charging_first_leg" or
            #         (self.charge_in_break_started and agv.scheduled_charging)):
                # if not self.charge_needed(False, agv.id):
            if not agv.servicing_order_type == "charging_first_leg":
                # if not self.charge_needed(False, agv.id):
                self.booked_idle_positions.remove(agv_pos)
                self.free_idle_positions.add(agv_pos)

        return agv

    def release_agv(self, position, system_time: float, agv_id: int):
        """
        Called on occurrence of second leg transport events to release the
        associated AGV.

        The released AGV is selected from the agv_index by means of the passed
        id. The the release is loged within the AGV object which in particular
        results in the update of the corresponding utilization time. The AGV
        is then added to the free_agv_positions index and the class counters
        are updated.

        :param position: The AGV position after the finished second leg
            transport.
        :param system_time: The current simulation time.
        :param agv_id: The id of the release AGV; note that the AGV ids are set
            on first leg Transport event creation and then passed to the
            corresponding second leg transport.
        :return: None.
        """
        released_agv = self.agv_index[agv_id]
        released_agv.log_release(system_time, position)
        if position in self.free_agv_positions\
                and released_agv.available_forks == 0:
            self.free_agv_positions[position].append(released_agv)
            self.n_busy_agvs -= 1
            self.n_free_agvs += 1
        elif released_agv.available_forks == 0:
            self.free_agv_positions[position] = [released_agv]
            self.n_busy_agvs -= 1
            self.n_free_agvs += 1
        elif released_agv.available_forks > 0:
            free_agvs_cp = self.free_agv_positions.copy()
            for agv in free_agvs_cp:
                index = 0
                for current_agv in free_agvs_cp[agv]:
                    if agv_id == current_agv.id:
                        self.free_agv_positions[agv].pop(index)
                        if len(self.free_agv_positions[agv]) == 0:
                            del self.free_agv_positions[agv]
                        if position in self.free_agv_positions:
                            self.free_agv_positions[position].append(
                                released_agv)
                        else:
                            self.free_agv_positions[position] = [released_agv]
                        break
                    index += 1
        released_agv.available_forks = released_agv.forks
        released_agv.servicing_order_type = None
        # assert agv_id in [v[0].id for k, v in self.free_agv_positions.items()]

    def charge_needed(self, loaded: bool, agv_id: int):
        # TODO: loaded / unloaded travel
        agv = self.agv_index[agv_id]
        max_duration = self.router.max_distance / self.router.speed

        if agv.battery <= max_duration * self.consumption_rate_unloaded:
            # charging required, travel can not be completed
            agv.charging_needed = True
            return True

    def charge_battery(self, charging_time: int, agv_id: int):
        agv = self.agv_index[agv_id]
        agv.battery += charging_time * self.charging_rate
        agv.charging_needed = False
        try:
            assert agv.battery <= 100
        except:
            print()

    def deplete_battery(self, t, loaded: bool, agv_id: int):
        agv = self.agv_index[agv_id]
        if loaded:
            consumption = self.consumption_rate_loaded * t
        else:
            consumption = self.consumption_rate_unloaded * t
        agv.battery -= consumption

    def get_charging_station(self, agv_position, core):
        if len(self.free_charging_stations) > 0:
            cs = self.get_close_charging_station(agv_position)
            self.free_charging_stations.remove(cs)
        else:
            cs = self.get_least_queued_charging_station()
        return cs

    def book_charging_station(self, cs: Tuple[int, int], agv: AGV):
        if cs in self.booked_charging_stations.keys():
            self.booked_charging_stations[cs].append(agv)
        else:
            self.booked_charging_stations[cs] = [agv]

    def release_charging_station(self, cs: Tuple[int, int], agv: AGV):
        idx = self.booked_charging_stations[cs].index(agv)
        self.booked_charging_stations[cs].pop(idx)
        if len(self.booked_charging_stations[cs]) == 0:
            self.free_charging_stations.add(cs)

    def get_close_charging_station(self, agv_position: Tuple[int, int]):
        selected_station = None
        min_distance = inf
        for cs_position in self.free_charging_stations:
            d = self.router.get_distance(agv_position, (cs_position[0],
                                                        cs_position[1]))
            if d < min_distance:
                selected_station = cs_position
                min_distance = d
        assert selected_station is not None
        return selected_station

    def get_least_queued_charging_station(self):
        charging_queue = self.queued_charging_events
        selected_station = None
        min_queued = inf
        for cs in charging_queue.keys():
            n = len(charging_queue[cs])
            if n < min_queued:
                selected_station = cs
                min_queued = n
        assert selected_station is not None
        return selected_station

    def get_close_idle_position(self, agv_position: Tuple[int, int]):
        assert (len(self.free_idle_positions) +
                len(self.booked_idle_positions) == self.n_agvs)
        selected_station = None
        min_distance = inf
        for idle_position in self.free_idle_positions:
            d = self.router.get_distance(agv_position, idle_position)
            if d < min_distance:
                selected_station = idle_position
                min_distance = d
        assert selected_station is not None
        self.free_idle_positions.remove(selected_station)
        self.booked_idle_positions.add(selected_station)
        return selected_station

    def get_close_agv(self, position: Tuple[int, int], order_type: str)\
            -> Tuple[int, int]:
        """
        Iterates over the free AGVs in the free_agv_positions and selects the
        one closest to the passed position with respect to euclidean distance.

        :param position: The position relative to which the closest AGV is to
            be selected.
        :return: The closest AGV position.
        """
        selected_agv = None
        min_distance = inf
        agv_index = 0
        for agv in self.free_agv_positions:
            # current_agv = self.free_agv_positions[agv][-1]
            index = 0
            for current_agv in self.free_agv_positions[agv]:
                if current_agv.servicing_order_type == order_type or\
                        current_agv.servicing_order_type is None and current_agv.free:
                    distance = hypot(position[0] - agv[0], position[1] - agv[1])
                    if distance < min_distance:
                        agv_index = index
                        selected_agv = agv
                        min_distance = distance
                index += 1
        # assert selected_agv in self.free_agv_positions
        return selected_agv, agv_index

    def get_agv_locations(self) -> Dict[Tuple[int, int], List[AGV]]:
        """
        Returns the free_agv_positions index.

        :return: The dictionary mapping free agv positions to lists of AGV
            objects.
        """
        return self.free_agv_positions

    def get_average_utilization(self):
        """
        Iterates over the agv_index and extracts the average utilization time.

        :return: The average utilization time.
        """
        utl_sum = 0
        for agv_id, agv in self.agv_index.items():
            utl_sum += agv.utilization
        return utl_sum / len(self.agv_index)

    def get_n_depleted_agvs(self) -> int:
        n_depleted = 0
        for agv in self.agv_index.keys():
            if self.agv_index[agv].charging_needed == True:
                n_depleted += 1
        return n_depleted

    def get_n_charged_agvs(self):
        n_depleted = 0
        for agv in self.agv_index.keys():
            if self.agv_index[agv].charging_needed == False:
                n_depleted += 1
        return n_depleted

    def get_n_charging_events(self):
        return self.agv_trackers.n_charges

    def mark_idle_position_arrival(self, idle_station, agv_id):
        # self.free_idle_positions.remove(idle_station)
        self.relocating_agvs.remove(agv_id)
        self.stationary_amr_positions.add(idle_station)

    def update_trackers_on_charging_end(self):
        self.agv_trackers.n_charges += 1
        total_battery = 0
        for agv in self.agv_index:
            total_battery += self.agv_index[agv].battery
        self.agv_trackers.avg_battery_level = total_battery / len(self.agv_index)

        for agv_id in self.agv_index.keys():
            self.agv_trackers.battery_level_per_agv[agv_id]\
                = self.agv_index[agv_id].battery
            self.agv_trackers.charges_per_agv[agv_id]\
                = self.agv_index[agv_id].n_charging_stops

        for cs in self.booked_charging_stations.keys():
            self.agv_trackers.queue_per_station[cs]\
                = len(self.booked_charging_stations[cs])


class AGVTrackers:
    def __init__(self, agvm: AgvManager):
        self.avg_battery_level = 0
        self.n_charges = 0
        self.queue_per_station = {cs: 0 for cs in agvm.free_charging_stations}
        self.battery_level_per_agv = {agv_id: 0 for agv_id in agvm.agv_index.keys()}
        self.charges_per_agv = {agv_id: 0 for agv_id in agvm.agv_index.keys()}
        self.orders_not_serviced_agvs_depleted = 0