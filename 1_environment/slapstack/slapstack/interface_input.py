from slapstack.interface_templates import SimulationParameters


class Input:
    def __init__(
            self, environment_parameters: SimulationParameters, seed: int = 1, partition: int = 0):
        self.params = environment_parameters
        self.seed = seed
        self.partition = partition
