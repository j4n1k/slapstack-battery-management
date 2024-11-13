import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
import pickle
from copy import deepcopy

import numpy as np
from numpy import genfromtxt
from slapstack import SlapEnv
from slapstack.core_state import State
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


class SLAPGurobiModel:
    def __init__(self,
                 time_periods: int,
                 locations: List[Tuple[int, int]],
                 deliveries: List[Dict],
                 retrievals: List[Dict],
                 agvs: int,
                 skus: int,
                 lanes: Dict[int, List[Tuple[int, int]]],
                 io_points: List[Tuple[int, int]],
                 initial_storage: Dict[Tuple[int, int], int],
                 travel_times: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
                 agv_initial_positions: List[Tuple[int, int]]):

        # Sets
        self.T = range(time_periods)
        self.I = locations
        self.D = range(len(deliveries))
        self.R = range(len(retrievals))
        self.K = range(agvs)
        self.S = range(skus)
        self.L = lanes
        self.N = io_points

        # Parameters
        self.deliveries = deliveries
        self.retrievals = retrievals
        self.travel_times = travel_times
        self.initial_storage = initial_storage
        self.agv_initial_positions = agv_initial_positions

        # Initialize model
        self.model = gp.Model("SLAP_Retrieval")

        # Build model
        self._create_variables()
        self._create_constraints()
        self._set_objective()

    def _create_variables(self):
        m = self.model

        # Main decision variables
        self.x = m.addVars([(i, r, k, t) for i in self.I
                            for r in self.R
                            for k in self.K
                            for t in self.T],
                           vtype=GRB.BINARY, name="retrieval")

        self.y = m.addVars([(i, d, k, t) for i in self.I
                            for d in self.D
                            for k in self.K
                            for t in self.T],
                           vtype=GRB.BINARY, name="delivery")

        # AGV position variables
        all_positions = self.I + self.N
        self.z = m.addVars([(i, k, t) for i in all_positions
                            for k in self.K
                            for t in self.T],
                           vtype=GRB.BINARY, name="position")

        # Movement tracking variables
        self.move = m.addVars([(i1, i2, k, t) for i1 in all_positions
                               for i2 in all_positions
                               for k in self.K
                               for t in range(len(self.T) - 1)],
                              vtype=GRB.BINARY, name="move")

        # Time variables
        self.f_d = m.addVars(self.D, name="delivery_completion")
        self.f_r = m.addVars(self.R, name="retrieval_completion")

        # Storage state
        self.p = m.addVars([(i, s, t) for i in self.I
                            for s in self.S
                            for t in self.T],
                           vtype=GRB.BINARY, name="storage")

    def _create_constraints(self):
        m = self.model
        all_positions = self.I + self.N

        # Order completion time constraints
        for d in self.D:
            arrival_time = self.deliveries[d]['arrival_time']
            for t in self.T:
                m.addConstr(
                    self.f_d[d] >= t * gp.quicksum(self.y[i, d, k, t]
                                                   for i in self.I
                                                   for k in self.K)
                )
            m.addConstr(self.f_d[d] >= arrival_time)

        for r in self.R:
            arrival_time = self.retrievals[r]['arrival_time']
            for t in self.T:
                m.addConstr(
                    self.f_r[r] >= t * gp.quicksum(self.x[i, r, k, t]
                                                   for i in self.I
                                                   for k in self.K)
                )
            m.addConstr(self.f_r[r] >= arrival_time)
        #
        # Order assignment constraints
        for d in self.D:
            m.addConstr(gp.quicksum(self.y[i, d, k, t]
                                    for i in self.I
                                    for k in self.K
                                    for t in self.T) == 1)

            for t in range(self.deliveries[d]['arrival_time']):
                m.addConstr(gp.quicksum(self.y[i, d, k, t]
                                        for i in self.I
                                        for k in self.K) == 0)

        for r in self.R:
            m.addConstr(gp.quicksum(self.x[i, r, k, t]
                                    for i in self.I
                                    for k in self.K
                                    for t in self.T) == 1)

            for t in range(self.retrievals[r]['arrival_time']):
                m.addConstr(gp.quicksum(self.x[i, r, k, t]
                                        for i in self.I
                                        for k in self.K) == 0)
        #
        # Storage state tracking
        for i in self.I:
            for s in self.S:
                for t in range(len(self.T) - 1):
                    m.addConstr(
                        self.p[i, s, t + 1] == self.p[i, s, t] +
                        gp.quicksum(self.y[i, d, k, t] for d in self.D for k in self.K
                                    if self.deliveries[d]['sku'] == s) -
                        gp.quicksum(self.x[i, r, k, t] for r in self.R for k in self.K
                                    if self.retrievals[r]['sku'] == s)
                    )

        # Can only retrieve existing SKUs
        for i in self.I:
            for r in self.R:
                for k in self.K:
                    for t in self.T:
                        m.addConstr(
                            self.x[i, r, k, t] <= gp.quicksum(
                                self.p[i, s, t] for s in self.S
                                if self.retrievals[r]['sku'] == s
                            )
                        )
        #
        # # AGV movement constraints
        for k in self.K:
            for t in self.T:
                m.addConstr(gp.quicksum(self.z[i, k, t] for i in all_positions) == 1)

        # Movement linearization
        for k in self.K:
            for t in range(len(self.T) - 1):
                for i1 in all_positions:
                    for i2 in all_positions:
                        m.addConstr(self.move[i1, i2, k, t] <= self.z[i1, k, t])
                        m.addConstr(self.move[i1, i2, k, t] <= self.z[i2, k, t + 1])
                        m.addConstr(
                            self.move[i1, i2, k, t] >=
                            self.z[i1, k, t] + self.z[i2, k, t + 1] - 1
                        )

        # Travel time constraints
        for k in self.K:
            for t in range(len(self.T) - 1):
                for i1 in all_positions:
                    for i2 in all_positions:
                        if (i1, i2) in self.travel_times:
                            travel_time = self.travel_times[i1, i2]
                            if t + travel_time < len(self.T):
                                m.addConstr(
                                    self.move[i1, i2, k, t] <=
                                    1 - gp.quicksum(
                                        self.z[i3, k, t + tau]
                                        for i3 in all_positions if i3 != i2
                                        for tau in range(1, travel_time)
                                    )
                                )

        # Task execution constraints
        for k in self.K:
            for t in self.T:
                m.addConstr(
                    gp.quicksum(self.x[i, r, k, t] for i in self.I for r in self.R) +
                    gp.quicksum(self.y[i, d, k, t] for i in self.I for d in self.D) <= 1
                )

    def _set_objective(self):
        delivery_service_time = (gp.quicksum(self.f_d[d] - self.deliveries[d]['arrival_time']
                                             for d in self.D) / len(self.D))
        retrieval_service_time = (gp.quicksum(self.f_r[r] - self.retrievals[r]['arrival_time']
                                              for r in self.R) / len(self.R))

        self.model.setObjective(delivery_service_time + retrieval_service_time, GRB.MINIMIZE)

    def optimize(self, time_limit=None, gap=None, threads=None):
        if time_limit:
            self.model.setParam('TimeLimit', time_limit)
        if gap:
            self.model.setParam('MIPGap', gap)
        if threads:
            self.model.setParam('Threads', threads)

        self.model.optimize()

        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            return self._extract_solution()
        return None

    def _extract_solution(self) -> Dict:
        solution = {
            'objective': self.model.objVal,
            'delivery_schedules': {},
            'retrieval_schedules': {},
            'agv_schedules': {},
            'storage_state': {}
        }

        # Extract delivery schedules
        for d in self.D:
            for i in self.I:
                for k in self.K:
                    for t in self.T:
                        if self.y[i, d, k, t].X > 0.5:
                            solution['delivery_schedules'][d] = {
                                'location': i,
                                'agv': k,
                                'time': t,
                                'completion_time': self.f_d[d].X
                            }

        # Extract retrieval schedules
        for r in self.R:
            for i in self.I:
                for k in self.K:
                    for t in self.T:
                        if self.x[i, r, k, t].X > 0.5:
                            solution['retrieval_schedules'][r] = {
                                'location': i,
                                'agv': k,
                                'time': t,
                                'completion_time': self.f_r[r].X
                            }

        # Extract AGV schedules
        for k in self.K:
            solution['agv_schedules'][k] = []
            for t in self.T:
                for i in (self.I + self.N):
                    if self.z[i, k, t].X > 0.5:
                        solution['agv_schedules'][k].append((i, t))

        # Extract storage state
        for t in self.T:
            solution['storage_state'][t] = {}
            for i in self.I:
                for s in self.S:
                    if self.p[i, s, t].X > 0.5:
                        solution['storage_state'][t][i] = s

        return solution


def create_small_example():
    """
    Creates a small example with:
    - 3x3 storage grid with I/O points
    - 2 AGVs
    - 2 SKUs
    - 2 delivery orders and 2 retrieval orders
    - 10 time periods
    """

    # Time periods
    time_periods = 10

    # Locations: (x,y) coordinates for storage locations
    locations = [
        (1, 1), (1, 2), (1, 3),  # First row of storage
        (2, 1), (2, 2), (2, 3),  # Second row
        (3, 1), (3, 2), (3, 3)  # Third row
    ]

    # I/O points
    io_points = [(0, 0), (0, 4)]  # Input at (0,0), Output at (0,4)

    # Initial AGV positions
    agv_initial_positions = [(0, 0), (0, 4)]  # AGVs start at I/O points

    # Lanes (simple definition - each column is a lane)
    lanes = {
        0: [(1, 1), (2, 1), (3, 1)],
        1: [(1, 2), (2, 2), (3, 2)],
        2: [(1, 3), (2, 3), (3, 3)]
    }

    # Initial storage (empty initially)
    initial_storage = {}

    # Travel times (Manhattan distance)
    travel_times = {}
    all_positions = locations + io_points
    for pos1 in all_positions:
        for pos2 in all_positions:
            if pos1 != pos2:
                travel_times[(pos1, pos2)] = 1 # (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    # Delivery orders
    deliveries = [
        {
            'arrival_time': 0,
            'sku': 0,
            'io_point': (0, 0)
        },
        {
            'arrival_time': 2,
            'sku': 1,
            'io_point': (0, 0)
        }
    ]

    # Retrieval orders
    retrievals = [
        {
            'arrival_time': 4,
            'sku': 0,
            'io_point': (0, 4)
        },
        {
            'arrival_time': 6,
            'sku': 1,
            'io_point': (0, 4)
        }
    ]

    return {
        'time_periods': time_periods,
        'locations': locations,
        'deliveries': deliveries,
        'retrievals': retrievals,
        'agvs': 2,
        'skus': 2,
        'lanes': lanes,
        'io_points': io_points,
        'initial_storage': initial_storage,
        'travel_times': travel_times,
        'agv_initial_positions': agv_initial_positions
    }

# params = SimulationParameters(
#         use_case="minislap",
#         use_case_n_partitions=1,
#         use_case_partition_to_use=0,
#         n_agvs=3,
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
#         charging_thresholds=[0, 100],
#         partition_by_day=False,
#         battery_capacity=10,
#         charge_during_breaks=False
#     )
#
# environment: SlapEnv = get_env(
#     sim_parameters=params,
#     log_frequency=1000,
#     nr_zones=3,
#     log_dir='./result_data_charging_cross',
#     logfile_name='lookahead_minislap'
# )
# state: State = environment.core_env.state
# orders = state.params.order_list
#
#
# unit_dist = 1.4
# use_case = "minislap"
# layout_path = f"../1_environment/slapstack/slapstack/use_cases/{use_case}/1_layout.csv"
# layout = genfromtxt(layout_path, delimiter=',')
# layout = layout.astype(int)
# shape = layout.shape
# if layout[0, 0] != -1:
#     layout[0, 0] = -1
#     data = np.transpose(layout)
# else:
#     data = np.delete(layout, shape[1] - 1, axis=1)
#
# rows, cols = layout.shape
# storage_positions = set()
# source_docks = set()
# sink_docks = set()
#
# # Process each cell in the layout
# for row in range(rows):
#     for col in range(cols):
#         value = layout[row, col]
#
#         # Calculate coordinates (origin at top-left corner)
#         x = col * unit_dist
#         y = row * unit_dist
#
#         if value == 0:
#             storage_positions.add((x, y))
#         elif value == -3:
#             source_docks.add((x, y))
#         elif value == -4:
#             sink_docks.add((x, y))
#
# for order in orders:
#     if order[0] == "delivery":
#         {"arrival_time": order[1]}
# lanes = state.location_manager.lane_manager.lane_index
# # Initialize model
# model = SLAPGurobiModel(
#     time_periods=100,
#     locations=[(coord[0], coord[1]) for coord in storage_positions],
#     deliveries=[{
#         'arrival_time': order.arrival_time,
#         'sku': order.sku,
#         'io_point': order.dock
#     } for order in orders],
#     retrievals=[{
#         'arrival_time': order.arrival_time,
#         'sku': order.sku,
#         'io_point': order.dock
#     } for order in retrieval_orders],
#     agvs=len(state.agv_manager.agvs),
#     skus=max_sku + 1,
#     lanes=state.get_lanes(),
#     io_points=state.get_io_points(),
#     initial_storage=state.get_storage_state(),
#     travel_times=state.agv_manager.router.distance_matrix,
#     agv_initial_positions=[agv.position for agv in state.agv_manager.agvs]
# )


if __name__ == "__main__":
    # Generate example data
    example_data = create_small_example()

    # Initialize model
    model = SLAPGurobiModel(**example_data)

    # Set some solver parameters
    model.model.setParam('TimeLimit', 60)  # 1 minute time limit
    model.model.setParam('MIPGap', 0.01)  # 1% optimality gap
    model.model.setParam('OutputFlag', 1)  # Show solver output

    # Optimize
    solution = model.optimize()

    if solution:
        print("\nSolution found!")
        print(f"Objective value (average service time): {solution['objective']:.2f}")

        print("\nDelivery Schedules:")
        for d, details in solution['delivery_schedules'].items():
            print(f"Delivery {d}: Location {details['location']} at time {details['time']} "
                  f"by AGV {details['agv']} (completion: {details['completion_time']:.1f})")

        print("\nRetrieval Schedules:")
        for r, details in solution['retrieval_schedules'].items():
            print(f"Retrieval {r}: Location {details['location']} at time {details['time']} "
                  f"by AGV {details['agv']} (completion: {details['completion_time']:.1f})")

        print("\nAGV Schedules:")
        for k, schedule in solution['agv_schedules'].items():
            print(f"AGV {k}: {schedule}")

        print("\nStorage State Over Time:")
        for t in range(example_data['time_periods']):
            if t in solution['storage_state']:
                print(f"Time {t}:")
                for loc, sku in solution['storage_state'][t].items():
                    print(f"  Location {loc}: SKU {sku}")
    else:
        print("No solution found!")
