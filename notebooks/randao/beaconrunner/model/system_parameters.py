from dataclasses import dataclass

from model.utils import default

@dataclass
class Parameters:
    frequency: int = default([1])
    network_update_rate: float = default([1.0])
    num_epochs: int = default([5])

parameters = Parameters().__dict__
