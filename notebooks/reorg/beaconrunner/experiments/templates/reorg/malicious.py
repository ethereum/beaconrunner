from model.specs import (
    config, SLOTS_PER_EPOCH, MAX_VALIDATORS_PER_COMMITTEE,
    Checkpoint, AttestationData, Attestation,
    BeaconBlock, BeaconBlockBody, SignedBeaconBlock,
    get_block_root_at_slot, compute_start_slot_at_epoch, process_slots,
    get_current_epoch, get_epoch_signature, process_block, get_head,
)
SECONDS_PER_SLOT = config.SECONDS_PER_SLOT
MIN_GENESIS_TIME = config.MIN_GENESIS_TIME
GENESIS_DELAY = config.GENESIS_DELAY

from model.network import (
    knowledge_set_union, knowledge_set
)

from model.validatorlib import (
    BRValidator,
    get_attestation_signature, aggregate_attestations, should_process_attestation,
    make_block,
)

from eth2spec.utils.ssz.ssz_typing import Container, List, uint64, Bitlist, Bitvector, Bytes32
from eth2spec.utils.ssz.ssz_impl import hash_tree_root


class MaliciousData:
    def __init__(
        self,
        malicious_validator_indices = [],
        malicious_block = None,
        latest_malicious_slot = None,
        malicious_attestations = [],
        n_recorded_attestations_before_attack = None,
        percentage_malicious_validators_attesting_forward = 0,
        n_malicious_validators_for_slot = 0,
        n_honest_validators_for_slot = 0,
    ):
        self.malicious_validator_indices = malicious_validator_indices
        self.latest_malicious_slot = latest_malicious_slot
        self.malicious_block = malicious_block
        self.malicious_attestations = malicious_attestations
        self.n_recorded_attestations_before_attack = n_recorded_attestations_before_attack
        self.percentage_malicious_validators_attesting_forward = percentage_malicious_validators_attesting_forward
        self.n_malicious_validators_for_slot = n_malicious_validators_for_slot
        self.n_honest_validators_for_slot = n_honest_validators_for_slot

    def reset_attack(self):
        self.latest_malicious_slot = None
        self.malicious_block = None
        self.malicious_attestations = []
        self.n_recorded_attestations_before_attack = None
        print("resetting attack")

    def should_release_private_items(self, sim_time_ms, known_items):
        # If we're not attacking, then no need to worry
        if self.latest_malicious_slot is None:
            return False

        # We proposed privately the block at slot n+1
        # So `self.latest_malicious_slot = n+1`
        # We wait for an honest validator to release the slot at n+2
        # and for honest validators to attest at slot n+2
        # Since a slot is `SECONDS_PER_SLOT` long, we need to release the items
        # `(1+1/3) * SECONDS_PER_SLOT` after the start of slot n+1
        # Let's just do `1.5 * SECONDS_PER_SLOT` to be sure...
        time_to_wait_ms = 1.5 * int(SECONDS_PER_SLOT) * 1000
        attack_start_ms = float(int(MIN_GENESIS_TIME) + int(GENESIS_DELAY) + int(self.latest_malicious_slot) * SECONDS_PER_SLOT) * 1000
        total_time = attack_start_ms + time_to_wait_ms

        # When the simulation time is after the delay
        return (sim_time_ms >= total_time)

def make_malicious_attestation(
    validator: BRValidator,
    known_items: dict,
    malicious_data: MaliciousData
) -> Attestation:

    # Unpacking
    validator_index = validator.validator_index
    store = validator.store
    committee_slot = validator.data.current_attest_slot
    committee_index = validator.data.current_committee_index
    committee = validator.data.current_committee

    if malicious_data.malicious_block is None:
        return None

    # What am I attesting for?
    signed_block = malicious_data.malicious_block
    validator.record_block(signed_block)
    block = signed_block.message
    store = validator.store
    block_root = hash_tree_root(block)
    head_state = store.block_states[block_root].copy()
    if head_state.slot < committee_slot:
        process_slots(head_state, committee_slot)
    start_slot = compute_start_slot_at_epoch(get_current_epoch(head_state))
    epoch_boundary_block_root = block_root if start_slot == head_state.slot else get_block_root_at_slot(head_state, start_slot)
    tgt_checkpoint = Checkpoint(epoch=get_current_epoch(head_state), root=epoch_boundary_block_root)

    att_data = AttestationData(
        index = committee_index,
        slot = committee_slot,
        beacon_block_root = block_root,
        source = head_state.current_justified_checkpoint,
        target = tgt_checkpoint
    )

    # Set aggregation bits to myself only
    committee_size = len(committee)
    index_in_committee = committee.index(validator_index)
    aggregation_bits = Bitlist[MAX_VALIDATORS_PER_COMMITTEE](*([0] * committee_size))
    aggregation_bits[index_in_committee] = True # set the aggregation bit of the validator to True
    attestation = Attestation(
        aggregation_bits=aggregation_bits,
        data=att_data
    )
    attestation_signature = get_attestation_signature(head_state, att_data, validator.privkey)
    attestation.signature = attestation_signature

    print(validator.validator_index, "(malicious) attesting for malicious block")

    return attestation

def make_malicious_block(
    validator: BRValidator,
    known_items: dict,
    malicious_data: MaliciousData
) -> SignedBeaconBlock:

    print(validator.validator_index, "(malicious) proposing block for slot", validator.data.slot)

    slot = validator.data.slot
    head = validator.data.head_root
    processed_state = validator.process_to_slot(head, slot)

    attestations = [att for att in known_items["attestations"] if should_process_attestation(processed_state, att.item)]
    attestations = aggregate_attestations([att.item for att in attestations if slot <= att.item.data.slot + SLOTS_PER_EPOCH])

    signed_block = make_block(slot, head, validator, processed_state, attestations)

    return signed_block

def get_wasted_attestations(
    malicious_data: MaliciousData,
    known_items: dict
) -> int:
    latest_malicious_slot = malicious_data.latest_malicious_slot

    if latest_malicious_slot == None:
        return 0

    else:
        n_recorded_attestations_before_attack = malicious_data.n_recorded_attestations_before_attack
        n_recorded_attestations_after_attack_start = len([att for att in known_items["attestations"] if att.item.data.slot >= latest_malicious_slot])

        return (n_recorded_attestations_after_attack_start - n_recorded_attestations_before_attack)

### State update blocks

def reset_attack(params, step, sL, s, _input):
    malicious_data = s['malicious_data']
    sim_time_ms = s["sim_time_ms"]
    network = s["network"]

    validator = network.validators[0]

    known_items = knowledge_set_union(network, malicious_data.malicious_validator_indices)

    # If it released the private data already, we can reset
    if malicious_data.should_release_private_items(sim_time_ms, known_items):
        malicious_data.reset_attack()

    return ('malicious_data', malicious_data)

def update_malicious_data_propose(params, step, sL, s, _input):
    malicious_data = s['malicious_data']
    network = s['network']
    blocks = _input['malicious_blocks']
    if len(blocks) == 0:
        return ('malicious_data', malicious_data)

    # We could have several malicious_blocks returned at the same time
    # But not in this attack, so let's take the first block of the list
    validators = network.validators
    validator = validators[0]

    current_slot = validator.data.slot

#     for validator in network.validators:
#         if(len(validator.data.recorded_attestations) > base):
#             base = len(validator.data.recorded_attestations)

    block = blocks[0]
    malicious_data.malicious_block = block
    malicious_data.latest_malicious_slot = block.message.slot

    known_items = knowledge_set_union(network, malicious_data.malicious_validator_indices)
    n_recorded_attestations_before_attack = len([att for att in known_items["attestations"] if att.item.data.slot < block.message.slot])

    malicious_data.n_recorded_attestations_before_attack = n_recorded_attestations_before_attack

    return ('malicious_data', malicious_data)

def update_malicious_data_attest(params, step, sL, s, _input):
    malicious_data = s['malicious_data']
    attestations = _input['malicious_attestations']
    malicious_data.malicious_attestations += attestations
    return ('malicious_data', malicious_data)

def update_sim_time(params, step, sL, s, _input):
    # Record the current simulation time, to make our lives easier

    network = s['network']
    validator = network.validators[0]
    return ("sim_time_ms", validator.data.time_ms)

def update_percentage_malicious_validators_attesting_forward(params, step, sL, s, _input):

    malicious_data = s['malicious_data']
    network = s['network']

    validators = network.validators
    validator = validators[0]

    current_slot = validator.data.slot
    # CONV: for "number" variables, preface with `n_`, e.g., `n_attesters_for_current_slot`
    # n_attesters_for_current_slot = len([v for v in validators if v.data.current_attest_slot == current_slot])
    n_attesters_for_current_slot = len([v for v in validators if v.data.current_attest_slot == current_slot])
    n_malicious_attesters_for_current_slot = len([v for v in validators if (v.data.current_attest_slot == current_slot) and (v.validator_behaviour == "malicious")])
    if(current_slot == SLOTS_PER_EPOCH):
        n_attesters_for_next_slot = len([v for v in validators if v.data.next_attest_slot == 0])
        n_malicious_attesters_for_next_slot = len([v for v in validators if (v.data.next_attest_slot == 0) and (v.validator_behaviour == "malicious")])
    else:
        n_attesters_for_next_slot = len([v for v in validators if v.data.current_attest_slot == current_slot + 1])
        n_malicious_attesters_for_next_slot = len([v for v in validators if (v.data.current_attest_slot == current_slot + 1) and (v.validator_behaviour == "malicious")])

    malicious_data.percentage_malicious_validators_attesting_forward = (n_malicious_attesters_for_current_slot + n_malicious_attesters_for_next_slot)/(n_attesters_for_current_slot + n_attesters_for_next_slot)

    malicious_data.n_malicious_validators_for_slot = n_malicious_attesters_for_current_slot
    malicious_data.n_honest_validators_for_slot = n_attesters_for_current_slot - n_malicious_attesters_for_current_slot

    return ('malicious_data', malicious_data)

### Malicious policies

def malicious_attest_policy(params, step, sL, s):
    # Pinging malicious validators to check if they want to maliciously attest

    network = s['network']
    malicious_data = s['malicious_data']
    produced_attestations = []

    for validator_index, validator in enumerate(network.validators):

        if not validator_index in malicious_data.malicious_validator_indices:
            continue

        known_items = knowledge_set(network, validator_index)
        attestation = validator.malicious_attest(known_items, malicious_data)
        if attestation is not None:
            produced_attestations.append([validator_index, attestation])

    return ({ 'malicious_attestations': produced_attestations })

def malicious_attest_release_policy(params, step, sL, s):
    # When are we releasing the private attestations?

    malicious_data = s["malicious_data"]
    sim_time_ms = s["sim_time_ms"]
    network = s["network"]

    validator = network.validators[0]

    known_items = knowledge_set_union(network, malicious_data.malicious_validator_indices)

    if not malicious_data.should_release_private_items(sim_time_ms, known_items):
        return { "attestations": [] }

    return { "attestations": malicious_data.malicious_attestations }

def malicious_propose_policy(params, step, sL, s):
    # Pinging malicious validators to check if they want to maliciously propose

    network = s['network']
    malicious_data = s['malicious_data']
    produced_blocks = []

    for validator_index, validator in enumerate(network.validators):

        if not validator_index in malicious_data.malicious_validator_indices:
            continue

        known_items = knowledge_set(network, validator_index)
        block = validator.malicious_propose(known_items, malicious_data)
        if block is not None:
            produced_blocks.append(block)

    return ({ 'malicious_blocks': produced_blocks })

def malicious_propose_release_policy(params, step, sL, s):
    # When are we releasing the privately proposed blocks?

    malicious_data = s["malicious_data"]
    sim_time_ms = s["sim_time_ms"]
    network = s["network"]

    validator = network.validators[0]

    known_items = knowledge_set_union(network, malicious_data.malicious_validator_indices)

    if not malicious_data.should_release_private_items(sim_time_ms, known_items):
        return { "blocks": [] }

    return { "blocks": [malicious_data.malicious_block] }
