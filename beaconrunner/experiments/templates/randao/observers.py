import numpy as np

import model.specs as specs
import model.eth2 as eth2

# Common observers

def current_slot(params, substep, state_history, previous_state, policy_input):
    return ("current_slot", previous_state["network"].validators[0].data.slot)

def current_epoch(params, substep, state_history, previous_state, policy_input):
    return ("current_epoch", previous_state["network"].validators[0].data.current_epoch)

def get_epoch_proposers(validator, epochs_ahead=0):
    current_slot = specs.get_current_slot(validator.store)
    current_head_root = validator.get_head()
    current_state = validator.store.block_states[current_head_root].copy()
    current_epoch = specs.get_current_epoch(current_state)
    start_slot = specs.compute_start_slot_at_epoch(current_epoch)
    start_state = current_state.copy() if start_slot == current_state.slot else \
        validator.store.block_states[specs.get_block_root(current_state, current_epoch)].copy()

    epoch_proposers = []

    for slot in range(epochs_ahead * specs.SLOTS_PER_EPOCH + start_slot, start_slot + (epochs_ahead+1) * specs.SLOTS_PER_EPOCH):
        if slot < start_state.slot:
            continue
        if start_state.slot < slot:
            specs.process_slots(start_state, slot)
        epoch_proposers.append(specs.get_beacon_proposer_index(start_state))
    return epoch_proposers

def get_current_proposer_indices(params, substep, state_history, previous_state, policy_input):
    current_epoch_proposers = get_epoch_proposers(validator=previous_state["network"].validators[0], epochs_ahead=0)
    return ("current_indices", current_epoch_proposers)

def get_next_proposer_indices(params, substep, state_history, previous_state, policy_input):
    next_epoch_proposers = get_epoch_proposers(validator=previous_state["network"].validators[0], epochs_ahead=1)
    return ("next_indices", next_epoch_proposers)

def get_plus_2_epoch_proposer_indices(params, substep, state_history, previous_state, policy_input):
    plus_2_epoch_proposers = get_epoch_proposers(validator=previous_state["network"].validators[0], epochs_ahead=2)
    return ("plus_2_indices", plus_2_epoch_proposers)

def balances(params, substep, state_history, previous_state, policy_input):
    validators = previous_state["network"].validators
    validator = validators[0]
    head = specs.get_head(validator.store)
    current_state = validator.store.block_states[head]
    current_epoch = specs.get_current_epoch(current_state)
    indices = [i for i, v in enumerate(validators)]
    balances = [b for i, b in enumerate(current_state.balances)]
    return ("balances", [np.round(eth2.gwei_to_eth(balance), 6) for balance in balances])
