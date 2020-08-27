import time

from typing import Set, Optional, Sequence, Tuple, Dict, Text
from dataclasses import dataclass, field

from .specs import (
    Slot, Root, Epoch, CommitteeIndex, ValidatorIndex, Store,
    BeaconState, BeaconBlock, BeaconBlockBody, SignedBeaconBlock,
    Attestation, AttestationData, Checkpoint, BLSSignature,
    MAX_VALIDATORS_PER_COMMITTEE, VALIDATOR_REGISTRY_LIMIT,
    SLOTS_PER_EPOCH, DOMAIN_RANDAO, DOMAIN_BEACON_PROPOSER,
    DOMAIN_BEACON_ATTESTER,
    get_forkchoice_store, get_current_slot, compute_epoch_at_slot,
    get_head, process_slots, on_tick, get_current_epoch,
    get_committee_assignment, compute_start_slot_at_epoch,
    get_block_root, process_block, process_attestation,
    get_block_root_at_slot, get_beacon_proposer_index,
    get_domain, compute_signing_root, state_transition,
    on_block, on_attestation,
)

import milagro_bls_binding as bls
from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.utils.ssz.ssz_typing import Container, List, uint64, Bitlist, Bytes32
from eth2spec.test.helpers.keys import pubkeys, pubkey_to_privkey

frequency = 1
assert frequency in [1, 10, 100, 1000]

class ValidatorMove(object):
    """
    Internal class recording validator moves: messages sent over the wire by the validator.
    Useful e.g. to avoid double-sending an attestation.
    """

    time: uint64
    """Simulation time (in ms) when move was made."""

    slot: Slot
    """Slot where move was made."""

    move: str
    """Type of move. Currently either 'attest' or 'propose'"""

    def __init__(self, time, slot, move):
        """Initialise ValidatorMove"""

        self.time = time
        self.slot = slot
        self.move = move

class ValidatorData:
    """
    Holds current validator data, to be consumed by BRValidator subclasses.
    """
    slot: Slot
    """Current slot"""

    time_ms: uint64
    """Current simulation time in milliseconds"""

    head_root: Root
    """Current head root, returned by `get_head` on validator's `Store`"""

    current_epoch: Epoch
    """Current epoch"""

    current_attest_slot: Slot
    """Last computed slot to attest in the current epoch"""

    current_committee_index: CommitteeIndex
    """Last computed committee index to attest in the current epoch"""

    current_committee: List[ValidatorIndex, MAX_VALIDATORS_PER_COMMITTEE]
    """Last computed committee to attest in the current epoch"""

    next_attest_slot: Slot
    """Last computed slot to attest in the next epoch"""

    next_committee_index: CommitteeIndex
    """Last computed committee index to attest in the next epoch"""

    next_committee: List[ValidatorIndex, MAX_VALIDATORS_PER_COMMITTEE]
    """Last computed committee to attest in the next epoch"""

    last_slot_attested: Optional[Slot]
    """
    Last slot where validator attested. Possibly we have
    `self.slot == self.last_slot_attested`
    """

    current_proposer_duties: Sequence[bool]
    """
    For the next SLOTS_PER_EPOCH, up to the beginning of a new epoch,
    is the validator a block proposer?
    """

    last_slot_proposed: Optional[Slot]
    """
    Last slot where validator proposed a block. Possibly we have
    `self.slot == self.last_slot_proposed`
    """

    recorded_attestations: List[Root, VALIDATOR_REGISTRY_LIMIT]
    """
    Hash roots of `Store` recorded attestations. Used for internal cache.
    """

    received_block: bool
    """
    Has the validator received a block for `self.slot`?
    """

class HashableSpecStore(Container):
    """ We cache a map from current state of the `Store` to `head`, since `get_head`
    is computationally intensive. But `Store` is not hashable right off the bat.
    `get_head` only depends on stored blocks and latest messages, so we use that here.
    """

    recorded_attestations: List[Root, VALIDATOR_REGISTRY_LIMIT]
    """Recorded attestations in the `Store`"""

    recorded_blocks: List[Root, VALIDATOR_REGISTRY_LIMIT]
    """Recorded blocks in the `Store`"""

class BRValidator:
    """
    Abstract superclass from which validator behaviours inherit.
    Defines and maintains environment accessor functions (is the validator an attester? proposer?)
    Performs caching to avoid recomputing expensive operations.

    In general, you are not expected to use any of the methods or attributes defined here, _except_
    for `validator.data`, which exposes current simulation environment properties, up-to-date with
    respect to the validator (e.g., proposer and attester duties).

    Subclasses of `BRValidator` must define at least two methods:

    - `attest(self, known_items) -> Optional[Attestation]`
    - `propose(self, known_items) -> Optional[Attestation]`
    """

    validator_index: ValidatorIndex
    """Validator index in the simulation."""

    pubkey: int
    """Validator public key."""

    privkey: int
    """Validator private key."""

    store: Store
    """`Store` objects are defined in the specs."""

    history: List[ValidatorMove, VALIDATOR_REGISTRY_LIMIT]
    """History of `ValidatorMove` by the validator."""

    data: ValidatorData
    """Current validator data. Maintained by the `BRValidator` methods."""

    head_store: Dict[Root, Root] = {}
    """
    Static cache for expensive operations.
    `head_store` stores a map from store hash to head root.
    """

    state_store: Dict[Tuple[Root, Slot], BeaconState] = {}
    """
    Static cache for expensive operations.
    `state_store` stores a map from `(current_state_hash, to_slot)` calling
    `process_slots(current_state, to_slot)`.
    """

    def __init__(self, validator_index: ValidatorIndex):
        """
        Validator constructor
        We preload a bunch of things, to be updated later on as needed
        The validator is initialised from some base state, and given a `validator_index`
        """

        self.validator_index = validator_index
        self.pubkey = pubkeys[validator_index]
        self.privkey = pubkey_to_privkey[self.pubkey].to_bytes(32, 'big')

        self.history = []

        self.data = ValidatorData()

    def load_state(self, state: BeaconState) -> None:
        """
        """

        self.store = get_forkchoice_store(state.copy())

        self.data.time_ms = self.store.time * 1000
        self.data.recorded_attestations = []
        self.data.slot = get_current_slot(self.store)
        self.data.current_epoch = compute_epoch_at_slot(self.data.slot)
        self.data.head_root = self.get_head()

        current_state = state.copy()
        if current_state.slot < self.data.slot:
            process_slots(current_state, self.data.slot)

        self.update_attester(current_state, self.data.current_epoch)
        self.update_proposer(current_state)
        self.update_data()

    def get_hashable_store(self) -> HashableSpecStore:
        """
        Returns a hash of the current store state.

        Args:
            self (BRValidator): Validator

        Returns:
            HashableSpecStore: A hashable representation of the current `self.store`
        """

        return HashableSpecStore(
            recorded_attestations = self.data.recorded_attestations,
            recorded_blocks = list(self.store.blocks.keys())
        )

    def get_head(self) -> Root:
        """
        Our cached reimplementation of specs-defined `get_head`.

        Args:
            self (BRValidator): Validator

        Returns:
            Root: Current head according to the validator `self.store`
        """

        store_root = hash_tree_root(self.get_hashable_store())

        # If we can get the head from the cache, great!
        if store_root in BRValidator.head_store:
            return BRValidator.head_store[store_root]

        # Otherwise we must compute it again :(
        else:
            head_root = get_head(self.store)
            BRValidator.head_store[store_root] = head_root
            return head_root

    def process_to_slot(self, current_head_root: Root, slot: Slot) -> BeaconState:
        """
        Our cached `process_slots` operation.

        Args:
            self (BRValidator): Validator
            current_head_root (Root): Process to slot from this state root
            slot (Slot): Slot to process to

        Returns:
            BeaconState: Post-state after transition to `slot`
        """

        # If we want to fast-forward a state root to some slot, we check if we have already recorded the
        # resulting state.
        if (current_head_root, slot) in BRValidator.state_store:
            return BRValidator.state_store[(current_head_root, slot)].copy()

        # If we haven't, we need to process it.
        else:
            current_state = self.store.block_states[current_head_root].copy()

            if current_state.slot < slot:
                process_slots(current_state, slot)

            BRValidator.state_store[(current_head_root, slot)] = current_state
            return current_state.copy()

    def update_time(self, frequency: uint64 = frequency) -> None:
        """
        Moving validators' clocks by one step.
        To keep it simple, we assume frequency is a power of ten.

        Args:
            self (BRValidator): Validator
            frequency (uint64): Simulation update rate

        Returns:
            None
        """

        self.data.time_ms = self.data.time_ms + int(1000 / frequency)
        if self.data.time_ms % 1000 == 0:
            # The store is updated each second in the specs
            on_tick(self.store, self.store.time + 1)

            # If a new slot starts, we update
            if get_current_slot(self.store) != self.data.slot:
                self.update_data()

    def forward_by(self, seconds: uint64, frequency: uint64 = frequency) -> None:
        """
        A utility method to forward the clock by a given number of seconds.
        Useful for exposition!

        Args:
            self (BRValidator): Validator
            seconds (uint64): Number of seconds to fast-forward by
            frequency (uint64): Simulation update rate

        Returns:
            None
        """

        number_ticks = seconds * frequency
        for i in range(number_ticks):
            self.update_time(frequency)

    def update_attester(self, current_state: BeaconState, epoch: Epoch) -> None:
        """
        This is a fairly expensive operation, so we try not to call it when we don't have to.
        Update attester duties for the `epoch`.
        This can be queried no earlier than two epochs before
        (e.g., learn about epoch e + 2 duties at epoch t).

        Args:
            self (BRValidator): Validator
            current_state (BeaconState): The state from which proposer duties are computed
            epoch (Epoch): Either `current_epoch` or `current_epoch + 1`

        Returns:
            None
        """

        current_epoch = get_current_epoch(current_state)

        # When is the validator scheduled to attest in `epoch`?
        (committee, committee_index, attest_slot) = get_committee_assignment(
            current_state,
            epoch,
            self.validator_index)
        if epoch == current_epoch:
            self.data.current_attest_slot = attest_slot
            self.data.current_committee_index = committee_index
            self.data.current_committee = committee
        elif epoch == current_epoch + 1:
            self.data.next_attest_slot = attest_slot
            self.data.next_committee_index = committee_index
            self.data.next_committee = committee

    def update_proposer(self, current_state: BeaconState) -> None:
        """
        This is a fairly expensive operation, so we try not to call it when we don't have to.
        Update proposer duties for the current epoch.
        We need to check for each slot of the epoch whether the validator is a proposer or not.

        Args:
            self (BRValidator): Validator
            current_state (BeaconState): The state from which proposer duties are computed

        Returns:
            None
        """

        current_epoch = get_current_epoch(current_state)

        start_slot = compute_start_slot_at_epoch(current_epoch)

        start_state = current_state.copy() if start_slot == current_state.slot else \
        self.store.block_states[get_block_root(current_state, current_epoch)].copy()

        current_proposer_duties = []
        for slot in range(start_slot, start_slot + SLOTS_PER_EPOCH):
            if slot < start_state.slot:
                current_proposer_duties += [False]
                continue

            if start_state.slot < slot:
                process_slots(start_state, slot)

            current_proposer_duties += [get_beacon_proposer_index(start_state) == self.validator_index]

        self.data.current_proposer_duties = current_proposer_duties

    def update_attest_move(self) -> None:
        """
        When was the last attestation by the validator?
        Updates `self.data.last_slot_attested`.

        Args:
            self (BRValidator): Validator

        Returns:
            None
        """

        slots_attested = sorted([log.slot for log in self.history if log.move == "attest"], reverse = True)
        self.data.last_slot_attested = None if len(slots_attested) == 0 else slots_attested[0]

    def update_propose_move(self) -> None:
        """
        When was the last block proposal by the validator?
        Updates `self.data.last_slot_proposed`.

        Args:
            self (BRValidator): Validator

        Returns:
            None
        """

        slots_proposed = sorted([log.slot for log in self.history if log.move == "propose"], reverse = True)
        self.data.last_slot_proposed = None if len(slots_proposed) == 0 else slots_proposed[0]

    def update_data(self) -> None:
        """
        The head may change if we recorded a new block/new attestation in the `store`.
        Attester/proposer responsibilities may change if head changes *and*
        canonical chain changes to further back from start current epoch.

        .. code-block:: txt

            ---x------
                \           x is fork point
                 -----

            In the following
              attester = attester responsibilities for current epoch
              proposer = proposer responsibilities for current epoch

            - If x after current epoch change
              (---|--x , | = start current epoch),
              proposer and attester don't change
            - If x between start of previous epoch and
              start of current epoch
              (--||--x---|-- , || = start previous epoch)
              proposer changes but not attester
            - If x before start of previous epoch
              (--x--||-----|----) both proposer and attester change

        Args:
            self (BRValidator): Validator

        Returns:
            None
        """

        slot = get_current_slot(self.store)
        new_slot = self.data.slot != slot

        # Current epoch in validator view
        current_epoch = compute_epoch_at_slot(slot)

        self.update_attest_move()
        self.update_propose_move()

        # Did the validator record a block for this slot?
        received_block = len([block for block_root, block in self.store.blocks.items() if block.slot == slot]) > 0

        if not new_slot:
            # It's not a new slot, we are here because a new block/attestation was received

            # Getting the current state, fast-forwarding from the head
            head_root = self.get_head()

            if self.data.head_root != head_root:
                # New head!
                lca = lowest_common_ancestor(
                    self.store, self.data.head_root, head_root)
                lca_epoch = compute_epoch_at_slot(lca.slot)

                if lca_epoch == current_epoch:
                    # do nothing
                    pass
                else:
                    current_state = self.process_to_slot(head_root, slot)
                    if lca_epoch == current_epoch - 1:
                        self.update_proposer(current_state)
                    else:
                        self.update_proposer(current_state)
                        self.update_attester(current_state, current_epoch)
                self.data.head_root = head_root

        else:
            # It's a new slot. We should update our proposer/attester duties
            # if it's also a new epoch. If not we do nothing.
            if self.data.current_epoch != current_epoch:
                current_state = self.process_to_slot(self.data.head_root, slot)

                # We need to check our proposer role for this new epoch
                self.update_proposer(current_state)

                # We need to check our attester role for this new epoch
                self.update_attester(current_state, current_epoch)

        self.data.slot = slot
        self.data.current_epoch = current_epoch
        self.data.received_block = received_block

    def log_block(self, item: SignedBeaconBlock) -> None:
        """
        Recording 'block proposal' move by the validator in its history.
        """

        self.history.append(ValidatorMove(
            time = self.data.time_ms,
            slot = item.message.slot,
            move = "propose"
        ))
        self.update_propose_move()

    def log_attestation(self, item: Attestation) -> None:
        """
        Recording 'attestation proposal' move by the validator in its history.
        """

        self.history.append(ValidatorMove(
            time = self.data.time_ms,
            slot = item.data.slot,
            move = "attest"
        ))
        self.update_attest_move()

    def record_block(self, item: SignedBeaconBlock) -> bool:
        """
        When a validator receives a block from the network, they call `record_block` to see
        whether they should record it.
        """

        # If we already know about the block, do nothing
        if hash_tree_root(item.message) in self.store.blocks:
            return False

        # Sometimes recording the block fails. Examples include:
        # - The block slot is not the current slot (we keep it in memory for later, when we check backlog)
        # - The block parent is not known
        try:
            state = self.process_to_slot(item.message.parent_root, item.message.slot)
            on_block(self.store, item, state = state)
        except AssertionError as e:
            return False

        # If attestations are included in the block, we want to record them
        for attestation in item.message.body.attestations:
            self.record_attestation(attestation)

        return True

    def record_attestation(self, item: Attestation) -> bool:
        """
        When a validator receives an attestation from the network,
        they call `record_attestation` to see whether they should record it.
        """

        att_hash = hash_tree_root(item)

        # If we have already seen this attestation, no need to go further
        if att_hash in self.data.recorded_attestations:
            return False

        # Sometimes recording the attestation fails. Examples include:
        # - The attestation is not for the current slot *PLUS ONE*
        #   (we keep it in memory for later, when we check backlog)
        # - The block root it is attesting for is not known
        try:
            on_attestation(self.store, item)
            self.data.recorded_attestations += [att_hash]
            return True
        except:
            return False

    def check_backlog(self, known_items: Dict[str, Sequence[Container]]) -> None:
        """
        Called whenever a new event happens on the network that might make a validator update
        their internals.
        We loop over known blocks and attestations to check whether we should record any
        that we might have discarded before, or just received.
        """

        recorded_blocks = 0
        for block in known_items["blocks"]:
            recorded = self.record_block(block.item)
            if recorded:
                recorded_blocks += 1

        recorded_attestations = 0
        for attestation in known_items["attestations"]:
            recorded = self.record_attestation(attestation.item)
            if recorded:
                recorded_attestations += 1

        # If we do record anything, update the internals.
        if recorded_blocks > 0 or recorded_attestations > 0:
            self.update_data()

def lowest_common_ancestor(store, old_head, new_head) -> Optional[BeaconBlock]:
    """
    Find the lowest common ancestor to `old_head` and `new_head` in `store`.
    In most cases, `old_head` is an ancestor to `new_head`.
    We sort of (loosely) optimise for this.
    """

    new_head_ancestors = [new_head]
    current_block = store.blocks[new_head]
    keep_searching = True
    while keep_searching:
        parent_root = current_block.parent_root
        parent_block = store.blocks[parent_root]
        if parent_root == old_head:
            return store.blocks[old_head]
        elif parent_block.slot == 0:
            keep_searching = False
        else:
            new_head_ancestors += [parent_root]
            current_block = parent_block

    # At this point, old_head wasn't an ancestor to new_head
    # We need to find old_head's ancestors
    current_block = store.blocks[old_head]
    keep_searching = True
    while keep_searching:
        parent_root = current_block.parent_root
        parent_block = store.blocks[parent_root]
        if parent_root in new_head_ancestors:
            return parent_block
        elif parent_block.slot == 0:
            print("return none")
            return None
        else:
            current_block = parent_block

### Attestation strategies

def get_attestation_signature(state: BeaconState, attestation_data: AttestationData, privkey: int) -> BLSSignature:
    domain = get_domain(state, DOMAIN_BEACON_ATTESTER, attestation_data.target.epoch)
    signing_root = compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)

def honest_attest(validator, known_items):
    """
    Returns an honest attestation from `validator`.

    Args:
        validator (BRValidator): The attesting validator
        known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)

    Returns:
        Attestation: The honest attestation
    """

    # Unpacking
    validator_index = validator.validator_index
    store = validator.store
    committee_slot = validator.data.current_attest_slot
    committee_index = validator.data.current_committee_index
    committee = validator.data.current_committee

    # What am I attesting for?
    block_root = validator.get_head()
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

    return attestation

### Aggregation helpers

def get_aggregate_signature(attestations: Sequence[Attestation]) -> BLSSignature:
    signatures = [attestation.signature for attestation in attestations]
    return bls.Aggregate(signatures)

def build_aggregate(attestations):
    """
    Given a set of attestations from the same slot, committee index and vote for
    same source, target and beacon block, return an aggregated attestation.
    """

    if len(attestations) == 0:
        return []

    aggregation_bits = Bitlist[MAX_VALIDATORS_PER_COMMITTEE](*([0] * len(attestations[0].aggregation_bits)))
    for attestation in attestations:
        validator_index_in_committee = attestation.aggregation_bits.index(1)
        aggregation_bits[validator_index_in_committee] = True

    aggregate_attestation = Attestation(
        aggregation_bits=aggregation_bits,
        data=attestations[0].data
    )
    aggregate_signature = get_aggregate_signature(attestations)
    aggregate_attestation.signature = aggregate_signature

    return aggregate_attestation

def aggregate_attestations(attestations):
    """
    Take in a set of attestations. Output aggregated attestations.
    """

    hashes = set([hash_tree_root(att.data) for att in attestations])
    return [build_aggregate(
        [att for att in attestations if att_hash == hash_tree_root(att.data)]
    ) for att_hash in hashes]

### Proposal strategies

def get_block_signature(state: BeaconState, block: BeaconBlock, privkey: int) -> BLSSignature:
    domain = get_domain(state, DOMAIN_BEACON_PROPOSER, compute_epoch_at_slot(block.slot))
    signing_root = compute_signing_root(block, domain)
    return bls.Sign(privkey, signing_root)

def get_epoch_signature(state: BeaconState, block: BeaconBlock, privkey: int) -> BLSSignature:
    domain = get_domain(state, DOMAIN_RANDAO, compute_epoch_at_slot(block.slot))
    signing_root = compute_signing_root(compute_epoch_at_slot(block.slot), domain)
    return bls.Sign(privkey, signing_root)

def should_process_attestation(state: BeaconState, attestation: Attestation) -> bool:
    try:
        process_attestation(state.copy(), attestation)
        return True
    except:
        return False

def honest_propose(validator, known_items):
    """
    Returns an honest block, using the current LMD-GHOST head and all known, aggregated, attestations.

    Args:
        validator (BRValidator): The proposing validator
        known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)

    Returns:
        SignedBeaconBlock: The honest proposed block.
    """

    print(validator.validator_index, "proposing block for slot", validator.data.slot)

    slot = validator.data.slot
    head = validator.data.head_root

    processed_state = validator.process_to_slot(head, slot)

    attestations = [att for att in known_items["attestations"] if should_process_attestation(processed_state, att.item)]
    attestations = aggregate_attestations([att.item for att in attestations if slot <= att.item.data.slot + SLOTS_PER_EPOCH])

    beacon_block = BeaconBlock(
        slot=slot,
        parent_root=head,
        proposer_index = validator.validator_index,
    )

    beacon_block_body = BeaconBlockBody(
        attestations=attestations
    )
    epoch_signature = get_epoch_signature(processed_state, beacon_block, validator.privkey)
    beacon_block_body.randao_reveal = epoch_signature

    beacon_block.body = beacon_block_body

    process_block(processed_state, beacon_block)
    state_root = hash_tree_root(processed_state)
    beacon_block.state_root = state_root

    block_signature = get_block_signature(processed_state, beacon_block, validator.privkey)
    signed_block = SignedBeaconBlock(message=beacon_block, signature=block_signature)

    return signed_block
