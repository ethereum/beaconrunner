import random

from model.network import (
    update_network, disseminate_attestations,
    disseminate_sync_committees,
    disseminate_block, knowledge_set,
)

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
        blocks = validator.propose(known_items)

        if isinstance(blocks, list): # There is more than one block proposed...
            for block in blocks:
                produced_blocks.append(block)
        else: # There is not more than one block proposed
            if blocks is not None: # Check if any block at all has been proposed
                block = blocks
                produced_blocks.append(block)

    return ({ 'blocks': produced_blocks })
