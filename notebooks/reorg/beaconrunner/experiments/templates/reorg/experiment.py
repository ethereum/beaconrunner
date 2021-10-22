import copy
import numpy as np

from experiments.default_experiment import experiment

import model.parts.observers as observers
import model.parts.basic_duties as duties

import experiments.templates.reorg.observers as reorg_observers
import experiments.templates.reorg.malicious as malicious

import model.simulator as simulator
from model.network import Network, NetworkSet
from model.validators.MaliciousValidator import MaliciousValidator
from model.validators.ASAPValidator import ASAPValidator

# Make a copy of the default experiment to avoid mutation
experiment = copy.deepcopy(experiment)

reorg_observers = {
    "current_slot": reorg_observers.current_slot,
    "total_balance_asap": reorg_observers.total_balance_asap,
    "total_balance_malicious": reorg_observers.total_balance_malicious,
    "base_reward": reorg_observers.get_base_reward,
    "block_proposer": reorg_observers.get_block_proposer,
    "block_proposer_balance": reorg_observers.get_block_proposer_balance,
    "head": reorg_observers.get_head,
    "malicious_block": reorg_observers.get_malicious_block,
    "current_validator_state": reorg_observers.get_current_validator_state,
    "percentage": reorg_observers.get_percentage_malicious_attesting,
    "malicious_number": reorg_observers.get_n_malicious_attestors,
    "honest_number": reorg_observers.get_n_honest_attestors,
    "behaviour": reorg_observers.get_block_proposer_behaviour,
#    "blocks_display": reorg_observers.get_blocks,
#    "base_attestations": reorg_observers.get_base_attestations
    "is_attacking": reorg_observers.is_attacking,
    "current_validator_state": reorg_observers.get_current_validator_state,
}

def create_initial_network():
    rng = np.random.default_rng(269)

    num_validators = 20
    num_malicious = int((6 * num_validators) / 10)

    # We sample the position on the p2p network of prudent validators randomly
    malicious_set = set(
        rng.choice(np.arange(num_validators), size=num_malicious, replace=True)
    )
    print("malicious_indices", malicious_set)

    validators = []
    malicious_validator_indices = []

    # Initiate validators
    for i in range(num_validators):
        if i in malicious_set:
            new_validator = MaliciousValidator(i)
            malicious_validator_indices.append(i)
        else:
            new_validator = ASAPValidator(i)
        validators.append(new_validator)

    malicious_data = malicious.MaliciousData(
        malicious_validator_indices=malicious_validator_indices
    )

    set_a = NetworkSet(validators=list(range(num_validators)))
    network_sets = list([set_a])

    # Create a genesis state
    (genesis_state, genesis_block) = simulator.get_genesis_state_block(validators)

    # Validators load the state
    [v.load_state(genesis_state.copy(), genesis_block.copy()) for v in validators]

    # We skip the genesis block
    simulator.skip_genesis_block(validators)

    network = Network(validators=validators, sets=network_sets)
    return network, malicious_data

reorg_psubs = [
    {
        'policies': {
            'action': malicious.malicious_attest_policy,
        },
        'variables': {
            'malicious_data': malicious.update_malicious_data_attest,
        }
    },
    {
        'policies': {
            'action': duties.attest_policy
        },
        'variables': {
            'network': duties.update_attestations
        }
    },
    {
        'policies': {
            'action': malicious.malicious_propose_policy,
        },
        'variables': {
            'malicious_data': malicious.update_malicious_data_propose,
        }
    },
    {
        'policies': {
            'action': duties.propose_policy
        },
        'variables': {
            'network': duties.update_blocks
        }
    },
    {
        'policies': {
        },
        'variables': {
            'network': duties.tick
        }
    },
    {
        'policies': {
        },
        'variables': {
            'sim_time_ms': malicious.update_sim_time
        }
    },
    {
        'policies': {
        },
        'variables': {
            'malicious_data': malicious.update_percentage_malicious_validators_attesting_forward
        }
    },
    {
        'policies': {
            'action': malicious.malicious_attest_release_policy,
        },
        'variables': {
            'network': duties.update_attestations,
        }
    },
    {
        'policies': {
            'action': malicious.malicious_propose_release_policy,
        },
        'variables': {
            'network': duties.update_blocks,
        }
    },
    {
        'policies': {
        },
        'variables': {
            'malicious_data': malicious.reset_attack,
        }
    },
]

def create_model():
    new_model = copy.deepcopy(experiment.simulations[0].model)

    network, malicious_data = create_initial_network()
    new_model.initial_state["network"] = network
    new_model.initial_state["malicious_data"] = malicious_data
    new_model.initial_state["sim_time_ms"] = 0

    new_model.state_update_blocks = reorg_psubs

    parameter_overrides = {
        "num_epochs": [4]
    }
    new_model.params.update(parameter_overrides)

    (observed_ic, observed_psubs) = observers.add_observers(
        new_model.initial_state,
        new_model.state_update_blocks,
        reorg_observers
    )

    new_model.initial_state = observed_ic
    new_model.state_update_blocks = observed_psubs

    return new_model

def create_simulation():
    new_simulation = copy.deepcopy(experiment.simulations[0])
    new_simulation.model = create_model()
    new_simulation.timesteps = simulator.get_timesteps(new_simulation)
    return new_simulation

experiment.simulations = [create_simulation()]
