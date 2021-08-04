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

def tick(params, step, sL, s, _input):
    '''
    We call tick to move the clock by one step (= a second if frequency is 1, a tenth of a second
    if frequency is 10 etc). When tick moves the clock past the start of a new slot, validators
    update their internals, checking for instance their new attester or proposer duties if this
    tick coincides with a new epoch.

    Whenever tick is called, we also check whether we want the network to update or not, by
    flipping a biased coin. By "updating the network", we mean "peers exchange messages". In the
    chain example above, with 4 validators arranged as 0 <-> 1 <-> 2 <-> 3, it takes two network
    updates for a message from validator 3 to reach validator 0 (when validator 3 sends their
    message, we assume that it reaches all their peers instantly).

    The update frequency of the network is represented by the network_update_rate simulation
    parameter, also in Hertz. A network_update_rate of 1 means that messages spread one step
    further on the network each second.
    '''

    frequency = params["frequency"] # How many times per second we update the simulation.
    network_update_rate = params["network_update_rate"] # How many steps do messages propagate per second

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

### Block proposal

def propose_policy(params, step, sL, s):
    # Pinging validators to check if anyone wants to propose a block
    network = s['network']
    produced_blocks = []

    # What scenario are we running: "honest", "skip", "slashable"?
    scenario = params["scenario"]

    for validator_index, validator in enumerate(network.validators):
        known_items = knowledge_set(network, validator_index) # Known attestations&blocks of ValidatorIndex. (attestation info required to aggregate and put into block!)
        blocks = validator.propose(known_items, scenario=scenario) # Check, if supposed to propose and if yes, propose!
        # # # Prior way of doing things:
        # if block is not None:
        #     produced_blocks.append(block)
        if isinstance(blocks, list): # There is more than one block proposed...
            for block in blocks:
                produced_blocks.append(block)
        else: # There is not more than one block proposed
            if blocks is not None: # Check if any block at all has been proposed
                block = blocks # Changing "grammar" here lol
                produced_blocks.append(block)

    return ({ 'blocks': produced_blocks })

### Simulator shell

class SimulationParameters:

    num_epochs: uint64
    num_run: uint64
    frequency: uint64
    network_update_rate: float
    scenario: str

    def __init__(self, obj):
        self.num_epochs = obj["num_epochs"]
        self.num_run = obj["num_run"]
        self.frequency = obj["frequency"]
        self.network_update_rate = obj["network_update_rate"]
        self.scenario = obj["scenario"]

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
                'action': attest_policy # Ping all validators and check if they want to attest
            },
            'variables': {
                'network': update_attestations # Send attestations to direct peers respectively
            }
        },
        {
            'policies': {
                'action': propose_policy # Propose block if supposed to
            },
            'variables': {
                'network': update_blocks # Send block to direct peers respectively.
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
    steps = int(num_slots * SECONDS_PER_SLOT * parameters.frequency)

    params = {
        "frequency": [parameters.frequency],
        "network_update_rate": [parameters.network_update_rate],
        "scenario": [parameters.scenario],
    }

    print("Scenario ", parameters.scenario.capitalize(), ": simulating", parameters.num_epochs, "epochs (", num_slots, "slots ) at frequency", parameters.frequency, "moves/second")
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
            'scenario': [parameters.scenario],
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
