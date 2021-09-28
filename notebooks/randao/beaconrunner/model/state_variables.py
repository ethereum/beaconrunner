from dataclasses import dataclass

from model.network import Network

@dataclass
class StateVariables:
    network: Network = None

# Initialize State Variables instance with default values
initial_state = StateVariables().__dict__
