import model.specs as specs

from eth2spec.utils.ssz.ssz_impl import hash_tree_root

def extract_state(s):
    validators = s["network"].validators
    validator = validators[1]
    head = specs.get_head(validator.store)
    current_state = validator.store.block_states[head].copy()
    return current_state

def current_slot(params, step, sL, s, _input):
    return ("current_slot", s["network"].validators[0].data.slot)

def total_balance_asap(params, step, sL, s, _input):
    validators = s["network"].validators
    current_state = extract_state(s)
    current_epoch = specs.get_current_epoch(current_state)
    asap_indices = [i for i, v in enumerate(validators) if v.validator_behaviour == "asap"]
    asap_balances = [b for i, b in enumerate(current_state.balances) if i in asap_indices]
    return ("total_balance_asap", sum(asap_balances))

def total_balance_malicious(params, step, sL, s, _input):
    validators = s["network"].validators
    current_state = extract_state(s)
    current_epoch = specs.get_current_epoch(current_state)
    malicious_indices = [i for i, v in enumerate(validators) if v.validator_behaviour == "malicious"]
    malicious_balances = [b for i, b in enumerate(current_state.balances) if i in malicious_indices]
    return ("total_balance_malicious", sum(malicious_balances))

def get_base_reward(params, step, sL, s, _input):
    current_state = extract_state(s)
    base_reward = specs.get_base_reward(current_state, 0)
    return ("base_reward", base_reward)

def get_block_proposer(params, step, sL, s, _input):
    current_state = extract_state(s)
    block_proposer = [v.validator_index for v in s["network"].validators if v.data.current_proposer_duties[s["current_slot"] % specs.SLOTS_PER_EPOCH]][0]
    return ("block_proposer", block_proposer)

def get_block_proposer_behaviour(params, step, sL, s, _input):
    current_state = extract_state(s)
    block_proposer = [v for v in s["network"].validators if v.data.current_proposer_duties[s["current_slot"] % specs.SLOTS_PER_EPOCH]][0]
    behaviour = block_proposer.validator_behaviour
    return ("behaviour", behaviour)

def is_attacking(params, step, sL, s, _input):
    current_state = extract_state(s)
    block_proposer = [v for v in s["network"].validators if v.data.current_proposer_duties[s["current_slot"] % specs.SLOTS_PER_EPOCH]][0]
    if(block_proposer.validator_behaviour != "malicious"):
        return ("is_attacking", False)

    return("is_attacking", block_proposer.attacking)

def get_block_proposer_balance(params, step, sL, s, _input):
    current_state = extract_state(s)
    block_proposer_balance = current_state.balances[s["block_proposer"]]
    return ("block_proposer_balance", block_proposer_balance)

def get_sync_committee(params, step, sL, s, _input):
    current_state = extract_state(s)
    current_epoch = specs.get_current_epoch(current_state)
    sync_committee = current_state.current_sync_committee
    sync_committee = specs.get_sync_committee_indices(current_state, current_epoch)
    return ("sync_committee", sync_committee)

def get_head(params, step, sL, s, _input):
    validators = s["network"].validators
    validator = validators[4]
    head = specs.get_head(validator.store).hex()[0:6]
    return ("head", head)

# def get_blocks(params, step, sL, s, _input):
#     blocks = s["network"].blocks
#     blocks_display = [hash_tree_root(block).hex()[0:6] for block in blocks]
#     return ('blocks_display', blocks_display)

def get_percentage_malicious_attesting(params, step, sL, s, _input):
    percentage = s['malicious_data'].percentage_malicious_validators_attesting_forward
    if percentage is None:
        return ('percentage', "None")

    return ('percentage', percentage)

def get_n_malicious_attestors(params, step, sL, s, _input):
    number = s['malicious_data'].n_malicious_validators_for_slot
    return ('malicious_number', number)

def get_n_honest_attestors(params, step, sL, s, _input):
    number = s['malicious_data'].n_honest_validators_for_slot
    return ('honest_number', number)

def get_malicious_block(params, step, sL, s, _input):
    malicious_block = s['malicious_data'].malicious_block
    if malicious_block is None:
        return ('malicious_block', "None")

    return ('malicious_block', hash_tree_root(malicious_block.message).hex()[0:6])

def get_current_validator_state(params, step, sL, s, _input):
    current_state = extract_state(s)
    current_validator_state = []
    for v in s["network"].validators:
        current_validator_state += [{
            "slot": v.data.slot,
            "validator_index": v.validator_index,
            "balance": current_state.balances[v.validator_index],
            "block_proposer": 1 if s["block_proposer"] == v.validator_index else 0,
            "attester": 1 if v.data.current_attest_slot == v.data.slot else 0,
            "sync_committee": len(v.data.current_sync_committee),
            "behaviour": v.validator_behaviour
        }]
    return ("current_validator_state", current_validator_state)
