from model.specs import (
    Deposit, DepositData, BeaconState, BeaconBlock,
    config, SLOTS_PER_EPOCH,
    initialize_beacon_state_from_eth1, upgrade_to_altair,
)
SECONDS_PER_SLOT = config.SECONDS_PER_SLOT
MIN_GENESIS_TIME = config.MIN_GENESIS_TIME

from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.utils.hash_function import hash
from model.eth2 import eth_to_gwei

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

def get_timesteps(simulation):
    num_slots = simulation.model.params["num_epochs"][0] * SLOTS_PER_EPOCH
    return int(num_slots * SECONDS_PER_SLOT * simulation.model.params["frequency"][0])
