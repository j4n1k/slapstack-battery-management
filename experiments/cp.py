from ortools.sat.python import cp_model
from typing import Dict, List, Tuple
import pprint

def create_minimal_example():
    """
    Creates a minimal example with:
    - 2x2 storage grid
    - 1 I/O point
    - 2 AGVs
    - 2 SKUs
    - 2 deliveries and 1 retrieval
    - 10 time periods

    Layout:
    I - - -
    - S S -
    - S S -
    """

    # Time periods
    time_periods = 10

    # Storage locations (2x2 grid)
    locations = [
        (1, 1), (1, 2),  # First row
        (2, 1), (2, 2)  # Second row
    ]

    # I/O points
    io_points = [(0, 0)]  # Single I/O point

    # Travel times (Manhattan distance)
    travel_times = {}
    all_positions = locations + io_points
    for pos1 in all_positions:
        for pos2 in all_positions:
            if pos1 != pos2:
                travel_times[(pos1, pos2)] = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
            'sku': 0,  # Retrieve first delivered SKU
            'io_point': (0, 0)
        }
    ]

    # Initial storage state (empty warehouse)
    initial_storage = {}

    return {
        'time_periods': time_periods,
        'locations': locations,
        'deliveries': deliveries,
        'retrievals': retrievals,
        'agvs': 2,
        'skus': 2,
        'io_points': io_points,
        'initial_storage': initial_storage,
        'travel_times': travel_times
    }


def visualize_solution(solution, locations, time_periods):
    """Visualize the solution state at each time period."""
    print("\nSolution found!")
    print(f"Objective value (total service time): {solution['objective']}")

    print("\nDelivery Schedules:")
    for d, details in solution['deliveries'].items():
        print(f"Delivery {d}: Location {details['location']} at time {details['time']} "
              f"by AGV {details['agv']}")

    print("\nRetrieval Schedules:")
    for r, details in solution['retrievals'].items():
        print(f"Retrieval {r}: Location {details['location']} at time {details['time']} "
              f"by AGV {details['agv']}")

    print("\nStorage State Over Time:")
    max_x = max(x for x, y in locations)
    max_y = max(y for x, y in locations)

    for t in range(time_periods):
        print(f"\nTime {t}:")
        # Create grid
        grid = [['.' for _ in range(max_y + 1)] for _ in range(max_x + 1)]

        # Fill with SKUs
        if t in solution['storage_state']:
            for loc, sku in solution['storage_state'][t].items():
                x, y = locations[loc]
                grid[x][y] = str(sku)

        # Print grid
        for row in grid:
            print(' '.join(row))

class SLAPCPModel:
    def __init__(self,
                 time_periods: int,
                 locations: List[Tuple[int, int]],
                 deliveries: List[Dict],
                 retrievals: List[Dict],
                 agvs: int,
                 skus: int,
                 io_points: List[Tuple[int, int]],
                 initial_storage: Dict[Tuple[int, int], int],
                 travel_times: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]):

        self.T = range(time_periods)
        self.I = range(len(locations))
        self.K = range(agvs)
        self.S = range(skus)
        self.deliveries = deliveries
        self.retrievals = retrievals
        self.model = cp_model.CpModel()

        self._create_variables()
        self._create_constraints()
        self._set_objective()

    def _create_variables(self):
        max_time = len(self.T)

        # Task variables
        self.delivery_time = {}
        self.delivery_loc = {}
        self.delivery_agv = {}

        for d in range(len(self.deliveries)):
            self.delivery_time[d] = self.model.NewIntVar(
                self.deliveries[d]['arrival_time'], max_time - 1, f'delivery_time_{d}')
            self.delivery_loc[d] = self.model.NewIntVar(0, len(self.I) - 1, f'delivery_loc_{d}')
            self.delivery_agv[d] = self.model.NewIntVar(0, len(self.K) - 1, f'delivery_agv_{d}')

        self.retrieval_time = {}
        self.retrieval_loc = {}
        self.retrieval_agv = {}

        for r in range(len(self.retrievals)):
            self.retrieval_time[r] = self.model.NewIntVar(
                self.retrievals[r]['arrival_time'], max_time - 1, f'retrieval_time_{r}')
            self.retrieval_loc[r] = self.model.NewIntVar(0, len(self.I) - 1, f'retrieval_loc_{r}')
            self.retrieval_agv[r] = self.model.NewIntVar(0, len(self.K) - 1, f'retrieval_agv_{r}')

        # Storage state
        self.storage = {}
        for i in self.I:
            for t in self.T:
                self.storage[i, t] = self.model.NewIntVar(-1, len(self.S) - 1, f'storage_{i}_{t}')

    def _create_constraints(self):
        # Initial storage state
        for i in self.I:
            self.model.Add(self.storage[i, 0] == -1)

        # Storage state transitions
        for i in self.I:
            for t in range(len(self.T) - 1):
                # Default is to maintain state
                self.model.Add(self.storage[i, t + 1] == self.storage[i, t])

                # Storage updates from deliveries
                for d in range(len(self.deliveries)):
                    delivery_here = self.model.NewBoolVar(f'delivery_{d}_at_{i}_{t}')
                    # Location matches
                    self.model.Add(self.delivery_loc[d] == i).OnlyEnforceIf(delivery_here)
                    # Time matches
                    self.model.Add(self.delivery_time[d] == t).OnlyEnforceIf(delivery_here)
                    # Update storage if delivery happens
                    self.model.Add(
                        self.storage[i, t + 1] == self.deliveries[d]['sku']
                    ).OnlyEnforceIf(delivery_here)
                    # Location must be empty for delivery
                    self.model.Add(self.storage[i, t] == -1).OnlyEnforceIf(delivery_here)

                # Storage updates from retrievals
                for r in range(len(self.retrievals)):
                    retrieval_here = self.model.NewBoolVar(f'retrieval_{r}_at_{i}_{t}')
                    # Location matches
                    self.model.Add(self.retrieval_loc[r] == i).OnlyEnforceIf(retrieval_here)
                    # Time matches
                    self.model.Add(self.retrieval_time[r] == t).OnlyEnforceIf(retrieval_here)
                    # Location becomes empty after retrieval
                    self.model.Add(self.storage[i, t + 1] == -1).OnlyEnforceIf(retrieval_here)
                    # Must have correct SKU for retrieval
                    self.model.Add(
                        self.storage[i, t] == self.retrievals[r]['sku']
                    ).OnlyEnforceIf(retrieval_here)

        # AGV constraints - no overlapping tasks
        for k in self.K:
            for t in self.T:
                # Count tasks at this time for this AGV
                agv_tasks = []

                for d in range(len(self.deliveries)):
                    task_here = self.model.NewBoolVar(f'agv_{k}_delivery_{d}_at_{t}')
                    self.model.Add(self.delivery_agv[d] == k).OnlyEnforceIf(task_here)
                    self.model.Add(self.delivery_time[d] == t).OnlyEnforceIf(task_here)
                    agv_tasks.append(task_here)

                for r in range(len(self.retrievals)):
                    task_here = self.model.NewBoolVar(f'agv_{k}_retrieval_{r}_at_{t}')
                    self.model.Add(self.retrieval_agv[r] == k).OnlyEnforceIf(task_here)
                    self.model.Add(self.retrieval_time[r] == t).OnlyEnforceIf(task_here)
                    agv_tasks.append(task_here)

                # At most one task per AGV per time
                self.model.Add(sum(agv_tasks) <= 1)

        # Task ordering
        for d1 in range(len(self.deliveries)):
            for d2 in range(d1 + 1, len(self.deliveries)):
                # Different times for different deliveries
                self.model.Add(self.delivery_time[d1] != self.delivery_time[d2])

        for r1 in range(len(self.retrievals)):
            for r2 in range(r1 + 1, len(self.retrievals)):
                # Different times for different retrievals
                self.model.Add(self.retrieval_time[r1] != self.retrieval_time[r2])

    def _set_objective(self):
        total_time = 0
        # Minimize completion times relative to arrival times
        for d in range(len(self.deliveries)):
            total_time += self.delivery_time[d] - self.deliveries[d]['arrival_time']
        for r in range(len(self.retrievals)):
            total_time += self.retrieval_time[r] - self.retrievals[r]['arrival_time']
        self.model.Minimize(total_time)

    def solve(self, time_limit=None):
        solver = cp_model.CpSolver()
        if time_limit:
            solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._extract_solution(solver)
        return None

    def _extract_solution(self, solver):
        solution = {
            'objective': solver.ObjectiveValue(),
            'deliveries': {},
            'retrievals': {},
            'storage_state': {}
        }

        # Extract deliveries
        for d in range(len(self.deliveries)):
            solution['deliveries'][d] = {
                'time': solver.Value(self.delivery_time[d]),
                'location': solver.Value(self.delivery_loc[d]),
                'agv': solver.Value(self.delivery_agv[d])
            }

        # Extract retrievals
        for r in range(len(self.retrievals)):
            solution['retrievals'][r] = {
                'time': solver.Value(self.retrieval_time[r]),
                'location': solver.Value(self.retrieval_loc[r]),
                'agv': solver.Value(self.retrieval_agv[r])
            }

        # Extract storage state
        for t in self.T:
            solution['storage_state'][t] = {}
            for i in self.I:
                val = solver.Value(self.storage[i, t])
                if val >= 0:  # Only store non-empty locations
                    solution['storage_state'][t][i] = val

        return solution

def main():
    # Create example data
    example_data = create_minimal_example()

    # Initialize and solve model
    model = SLAPCPModel(**example_data)
    solution = model.solve(time_limit=10)  # 10 second time limit

    if solution:
        visualize_solution(solution, example_data['locations'], example_data['time_periods'])
    else:
        print("No solution found!")


if __name__ == "__main__":
    main()