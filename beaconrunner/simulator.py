import secrets
import time
import random
import pandas as pd

from typing import Dict, Callable, Any

# cadCAD configuration modules
from cadCAD.configuration.utils import config_sim
from cadCAD.configuration import Experiment

# cadCAD simulation engine modules
from cadCAD.engine import ExecutionMode, ExecutionContext
from cadCAD.engine import Executor

from cadCAD import configs
del configs[:]

from .specs import (
    Deposit, DepositData, BeaconState, BeaconBlock,
    SECONDS_PER_SLOT, SLOTS_PER_EPOCH, MIN_GENESIS_TIME,
    initialize_beacon_state_from_eth1, upgrade_to_altair,
)
from .network import (
    Network,
    update_network, disseminate_attestations,
    disseminate_sync_committees,
    disseminate_block, knowledge_set,
)

from .utils.cadCADsupSUP import (
    get_observed_psubs,
    get_observed_initial_conditions,
    get_observed_params,
    add_loop_ic,
    add_loop_psubs,
    add_loop_params,
    print_time,
)

from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.utils.ssz.ssz_typing import Bitlist, uint64
from eth2spec.utils.hash_function import hash
from .utils.eth2 import eth_to_gwei

## Initialisation

def get_initial_deposits(validators):
    """Produce deposits

    Args:
        validators (Sequence[BRValidator]): Validators of the simulation

    Returns:
        List[Deposit]: The list of deposits
    """

    return [Deposit(
        data=DepositData(
        amount=eth_to_gwei(32),
        pubkey=v.pubkey)
    ) for v in validators]

def get_genesis_state_block(validators, seed="hello"):
    block_hash = hash(seed.encode("utf-8"))
    eth1_timestamp = MIN_GENESIS_TIME
#     genesis_state = upgrade_to_altair(initialize_beacon_state_from_eth1(
#         block_hash, eth1_timestamp, get_initial_deposits(validators)
#     ))
    genesis_state = initialize_beacon_state_from_eth1(
        block_hash, eth1_timestamp, get_initial_deposits(validators)
    )
    genesis_block = BeaconBlock(state_root=hash_tree_root(genesis_state))
    return (genesis_state, genesis_block)

def skip_genesis_block(validators):
    for validator in validators:
        validator.forward_by(SECONDS_PER_SLOT)

## State transitions

def tick(params, step, sL, s, _input):
    # Move the simulation by one step

    frequency = params["frequency"]
    network_update_rate = params["network_update_rate"]

    # Probably overkill
    assert frequency >= network_update_rate

    network = s["network"]

    update_prob = float(network_update_rate) / float(frequency)

    # If we draw a success, based on `update_prob`, update the network
    if random.random() < update_prob:
        update_network(network)

    # Push validators' clocks by one step
    for validator in network.validators:
        validator.update_time(frequency)

    if s["timestep"] % 100 == 0:
        print("timestep", s["timestep"], "of run", s["run"])

    return ("network", network)

def update_attestations(params, step, sL, s, _input):
    # Get the attestations and disseminate them on-the-wire

    network = s["network"]
    disseminate_attestations(network, _input["attestations"])

    return ('network', network)

def update_sync_committees(params, step, sL, s, _input):
    # Get the sync committees and disseminate them on-the-wire

    network = s["network"]
    disseminate_sync_committees(network, _input["sc_bundles"])

    return ('network', network)

def update_blocks(params, step, sL, s, _input):
    # Get the blocks proposed and disseminate them on-the-wire

    network = s["network"]
    for block in _input["blocks"]:
        disseminate_block(network, block.message.proposer_index, block)

    return ('network', network)

## Policies

### Attestations

def attest_policy(params, step, sL, s):
    # Pinging validators to check if anyone wants to attest

    network = s['network']
    produced_attestations = []

    for validator_index, validator in enumerate(network.validators):
        known_items = knowledge_set(network, validator_index)
        attestation = validator.attest(known_items)
        if attestation is not None:
            produced_attestations.append([validator_index, attestation])

    return ({ 'attestations': produced_attestations })

### Sync aggregates proposal

def sync_committee_policy(params, step, sL, s):
    # Pinging validators to check if anyone wants to produce a sync committee

    network = s['network']
    produced_sc_bundles = []

    for validator_index, validator in enumerate(network.validators):
        known_items = knowledge_set(network, validator_index)
        sc_bundles = validator.sync_committees(known_items)
        if sc_bundles is not None:
            for sc_bundle in sc_bundles:
                produced_sc_bundles.append([validator_index, sc_bundle])

    return ({ 'sc_bundles': produced_sc_bundles })

### Block proposal

def propose_policy(params, step, sL, s):
    # Pinging validators to check if anyone wants to propose a block

    network = s['network']
    produced_blocks = []

    for validator_index, validator in enumerate(network.validators):
        known_items = knowledge_set(network, validator_index)
        block = validator.propose(known_items)
        if block is not None:
            produced_blocks.append(block)

    return ({ 'blocks': produced_blocks })

### Simulator shell

class SimulationParameters:

    num_epochs: uint64
    run_index: uint64
    frequency: uint64
    network_update_rate: float

    def __init__(self, obj):
        self.num_epochs = obj["num_epochs"]
        self.run_index = obj["run_index"]
        self.frequency = obj["frequency"]
        self.network_update_rate = obj["network_update_rate"]

def simulate(network: Network, parameters: SimulationParameters, observers: Dict[str, Callable[[BeaconState], Any]] = {}) -> pd.DataFrame:
    """
    Args:
        network (Network): Network of :py:class:`beaconrunner.validatorlib.BRValidator`
        parameters (BRSimulationParameters): Simulation parameters

    Returns:
        pandas.DataFrame: Results of the simulation contained in a pandas data frame
    """

    initial_conditions = {
        'network': network
    }

    psubs = [
        {
            'policies': {
                'action': attest_policy
            },
            'variables': {
                'network': update_attestations
            }
        },
        {
            'policies': {
                'action': sync_committee_policy
            },
            'variables': {
                'network': update_sync_committees
            }
        },
        {
            'policies': {
                'action': propose_policy
            },
            'variables': {
                'network': update_blocks
            }
        },
        {
            'policies': {
            },
            'variables': {
                'network': tick
            }
        },
    ]

    # Determine how many steps the simulation is running for
    num_slots = parameters.num_epochs * SLOTS_PER_EPOCH
    steps = int(num_slots * SECONDS_PER_SLOT * parameters.frequency)

    params = {
        "frequency": [parameters.frequency],
        "network_update_rate": [parameters.network_update_rate],
    }

    print("will simulate", parameters.num_epochs, "epochs (", num_slots, "slots ) at frequency", parameters.frequency, "moves/second")
    print("total", steps, "simulation steps")

    # Add our observers to the simulation
    observed_ic = get_observed_initial_conditions(initial_conditions, observers)
    observed_psubs = get_observed_psubs(psubs, observers)
    # observed_params = add_loop_params(get_observed_params(params, observers))

    sim_config = config_sim({
        'T': range(steps),
        'N': 1,
        'M': {
            'frequency': [parameters.frequency],
            'network_update_rate': [parameters.network_update_rate],
        }
    })

    from cadCAD import configs
    del configs[:]

    # Final simulation parameters and execution
    experiment = Experiment()
    experiment.append_configs(
        initial_state = observed_ic,
        partial_state_update_blocks = observed_psubs,
        sim_configs = sim_config
    )

    exec_context = ExecutionContext()
    simulation = Executor(exec_context=exec_context, configs=configs)
    raw_result, tensor, sessions = simulation.execute()

    return pd.DataFrame(raw_result)
