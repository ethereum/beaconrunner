import secrets
import time
import random
import pandas as pd

from typing import Dict, Callable, Any

from cadCAD.configuration import Configuration
from cadCAD.configuration.utils import config_sim
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor

from .specs import (
    Deposit, DepositData, BeaconState,
    SECONDS_PER_SLOT, SLOTS_PER_EPOCH,
    initialize_beacon_state_from_eth1,
)
from .network import (
    Network,
    update_network, disseminate_attestations,
    disseminate_block, knowledge_set,
)

from .utils.cadCADsupSUP import (
    get_observed_psubs,
    get_observed_initial_conditions,
    add_loop_ic,
    add_loop_psubs,
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

def get_genesis_state(validators, seed="hello"):
    block_hash = hash(seed.encode("utf-8"))
    eth1_timestamp = 1578009600
    return initialize_beacon_state_from_eth1(
        block_hash, eth1_timestamp, get_initial_deposits(validators)
    )

def skip_genesis_block(validators):
    for validator in validators:
        validator.forward_by(SECONDS_PER_SLOT)

## State transitions

def tick(_params, step, sL, s, _input):
    # Move the simulation by one step
    frequency = _params[0]["frequency"]
    network_update_rate = _params[0]["network_update_rate"]

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

def update_attestations(_params, step, sL, s, _input):
    # Get the attestations and disseminate them on-the-wire
    network = s["network"]
    disseminate_attestations(network, _input["attestations"])

    return ('network', network)

def update_blocks(_params, step, sL, s, _input):
    # Get the blocks proposed and disseminate them on-the-wire

    network = s["network"]
    for block in _input["blocks"]:
        disseminate_block(network, block.message.proposer_index, block)

    return ('network', network)

## Policies

### Attestations

def attest_policy(_params, step, sL, s):
    # Pinging validators to check if anyone wants to attest

    network = s['network']
    produced_attestations = []

    for validator_index, validator in enumerate(network.validators):
        known_items = knowledge_set(network, validator_index)
        attestation = validator.attest(known_items)
        if attestation is not None:
            produced_attestations.append([validator_index, attestation])

    return ({ 'attestations': produced_attestations })

### Block proposal

def propose_policy(_params, step, sL, s):
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
    num_run: uint64
    frequency: uint64
    network_update_rate: float

    def __init__(self, obj):
        self.num_epochs = obj["num_epochs"]
        self.num_run = obj["num_run"]
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
                'action': attest_policy # step 1
            },
            'variables': {
                'network': update_attestations # step 2
            }
        },
        {
            'policies': {
                'action': propose_policy # step 3
            },
            'variables': {
                'network': update_blocks # step 4
            }
        },
        {
            'policies': {
            },
            'variables': {
                'network': tick # step 5
            }
        },
    ]

    # Determine how many steps the simulation is running for
    num_slots = parameters.num_epochs * SLOTS_PER_EPOCH
    steps = num_slots * SECONDS_PER_SLOT * parameters.frequency

    simulation_parameters = {
        'T': range(steps),
        'N': 1,
        'M': {
            "frequency": [parameters.frequency],
            "network_update_rate": [parameters.network_update_rate],
        }
    }

    print("will simulate", parameters.num_epochs, "epochs (", num_slots, "slots ) at frequency", parameters.frequency, "moves/second")
    print("total", steps, "simulation steps")

    # Add our observers to the simulation
    observed_ic = add_loop_ic(get_observed_initial_conditions(initial_conditions, observers))
    observed_psubs = add_loop_psubs(get_observed_psubs(psubs, observers))

    # Final simulation parameters and execution
    configs = []
    for sim_param in config_sim(simulation_parameters):
        config = Configuration(sim_param,
                               initial_state=observed_ic,
                               partial_state_update_blocks=observed_psubs)
        configs.append(config)

    exec_mode = ExecutionMode()
    single_proc_ctx = ExecutionContext(context=exec_mode.single_proc)
    run = Executor(exec_context=single_proc_ctx, configs=configs)
    raw_result, tensor_field = run.execute()

    return pd.DataFrame(raw_result).assign(run = parameters.num_run)
