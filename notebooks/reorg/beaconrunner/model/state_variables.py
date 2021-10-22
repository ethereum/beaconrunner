from dataclasses import dataclass

from model.network import Network, NetworkSet
from model.validators.ASAPValidator import ASAPValidator
import model.simulator as simulator

def default_validators(num_validators=12):
    return [ASAPValidator(validator_index=i) for i in range(num_validators)]

def create_default_initial_network(
    validators=default_validators(),
    seed=""
) -> Network:
    num_validators = len(validators)
    set_a = NetworkSet(validators=list(range(num_validators)))
    network_sets = list([set_a])

    # Create a genesis state
    (genesis_state, genesis_block) = simulator.get_genesis_state_block(validators, seed=seed)

    # Validators load the state
    [v.load_state(genesis_state.copy(), genesis_block.copy()) for v in validators]

    # We skip the genesis block
    simulator.skip_genesis_block(validators)

    network = Network(validators=validators, sets=network_sets)
    return network

@dataclass
class StateVariables:
    network: Network = create_default_initial_network()

# Initialize State Variables instance with default values
initial_state = StateVariables().__dict__
