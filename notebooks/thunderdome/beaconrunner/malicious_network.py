import time
from typing import Set, Optional, Sequence, Tuple, Dict
from dataclasses import dataclass, field

from .specs import (
    VALIDATOR_REGISTRY_LIMIT,
    ValidatorIndex, Slot,
    BeaconState, Attestation, SignedBeaconBlock,
)
from .validatorlib import BRValidator

from eth2spec.utils.ssz.ssz_typing import Container, List, uint64

log = False # set to True to receive an avalanche of messages

class MaliciousNetworkSetIndex(uint64):
    pass

@dataclass
class MaliciousNetworkSet(object):
    validators: List[MaliciousNetworkSetIndex, (3/10)*VALIDATOR_REGISTRY_LIMIT]

@dataclass
class MaliciousNetworkAttestation(object):
    item: Attestation
    info_sets: List[MaliciousNetworkSetIndex, (3/10)*VALIDATOR_REGISTRY_LIMIT]

@dataclass
class MaliciousNetworkBlock(object):
    item: SignedBeaconBlock
    info_sets: List[MaliciousNetworkSetIndex, (3/10)*VALIDATOR_REGISTRY_LIMIT]

@dataclass
class MaliciousNetwork(object):
    validators: List[BRValidator, (3/10)*VALIDATOR_REGISTRY_LIMIT]
    sets: List[MaliciousNetworkSetIndex, (3/10)*VALIDATOR_REGISTRY_LIMIT]

    # In a previous implementation, we kept attestations and blocks in the same queue.
    # This was unwieldy. We can extend this easily by adding `Attester/ProposerSlashing`s
    attestations: List[MaliciousNetworkAttestation, (3/10)*VALIDATOR_REGISTRY_LIMIT] = field(default_factory=list)
    blocks: List[MaliciousNetworkBlock, (3/10)*VALIDATOR_REGISTRY_LIMIT] = field(default_factory=list)

    # We have the possibility of malicious validators refusing to propagate messages.
    # Unused so far and untested too.
    #malicious: List[ValidatorIndex, VALIDATOR_REGISTRY_LIMIT] = field(default_factory=list)

def get_all_sets_for_validator_MR(network: MaliciousNetwork, validator_index: ValidatorIndex) -> Sequence[MaliciousNetworkSetIndex]:
    # Return indices of sets to which the validator belongs

    return [i for i, s in enumerate(network.sets) if validator_index in s.validators]

def knowledge_set_MR(network: MaliciousNetwork, validator_index: ValidatorIndex) -> Dict[str, Sequence[Container]]:
    # Known attestations and blocks of validator `validator_index`

    info_sets = set(get_all_sets_for_validator_MR(network, validator_index))
    known_attestations = [item for item in network.attestations if len(set(item.info_sets) & info_sets) > 0]
    known_blocks = [item for item in network.blocks if len(set(item.info_sets) & info_sets) > 0]
    return { "attestations": known_attestations, "blocks": known_blocks }

def ask_to_check_backlog_malicious(network: MaliciousNetwork,
                         validator_indices: Set[ValidatorIndex]) -> None:
    # Called right after a message (block or attestation) was sent to `validator_indices`
    # Asks validators to check if they can e.g., definitely include attestations in their
    # latest messages or record blocks.
    for validator_index in validator_indices:
        validator = network.validators[validator_index]

        # Check if there are pending attestations/blocks that can be recorded
        known_items = knowledge_set_MR(network, validator_index)
        validator.check_backlog(known_items)

def disseminate_block(network: MaliciousNetwork,
                      sender: ValidatorIndex,
                      item: SignedBeaconBlock,
                      bignetwork: Network,
                      to_sets: List[MalicousNetworkSetIndex, (3/10)*VALIDATOR_REGISTRY_LIMIT] = None,
                      to_sets_big: List[NetworkSetIndex, VALIDATOR_REGISTRY_LIMIT] = None) -> None:
    # `sender` disseminates a block to its information sets, i.e., other validators they are peering
    # with.

    # Getting all the sets that `sender` belongs to
    broadcast_list = get_all_sets_for_validator_MR(network, sender) if to_sets is None else to_sets

    # The validator records that they have sent a block
    network.validators[sender].log_block(item)

    # Adding the block to network items
    networkItem = MaliciousNetworkBlock(item=item, info_sets=broadcast_list)
    network.blocks.append(networkItem)

    # A set of all validators who need to update their internals after reception of the block
    broadcast_validators = set()
    for info_set_index in broadcast_list:
        broadcast_validators |= set(network.sets[info_set_index].validators)

    ask_to_check_backlog_MR(network, broadcast_validators)

    ###Sleep(NextSlot + 4 - currentTime)

    broadcast_list = get_all_sets_for_validator(bignetwork, sender) if to_sets is None else to_sets

    # Adding the block to network items
    networkItem = NetworkBlock(item=item, info_sets=broadcast_list)
    bignetwork.blocks.append(networkItem)

    # A set of all validators who need to update their internals after reception of the block
    broadcast_validators = set()
    for info_set_index in broadcast_list:
        broadcast_validators |= set(bignetwork.sets[info_set_index].validators)

    ask_to_check_backlog(bignetwork, broadcast_validators)


def disseminate_attestations(network: MaliciousNetwork, items: Sequence[Tuple[ValidatorIndex, Attestation]], bignetwork: Network) -> None:
    # We get a set of attestations and disseminate them over the network

    # Finding out who receives a new attestation
    broadcast_validators = set()
    for item in items:
        sender = item[0]
        attestation = item[1]
        broadcast_list = get_all_sets_for_validator_MR(network, sender)

        # The sender records that they have sent an attestation
        network.validators[sender].log_attestation(attestation)

        # Adding the attestation to network items
        networkItem = MaliciousNetworkAttestation(item=attestation, info_sets=broadcast_list)
        network.attestations.append(networkItem)

        # Update list of validators who received a new item
        for info_set_index in broadcast_list:
            broadcast_validators |= set(network.sets[info_set_index].validators)

    ask_to_check_backlog_malicious(network, broadcast_validators)

    ###Sleep(NextSlot + 4 - currentTime)

    broadcast_validators = set()
    for item in items:
        sender = item[0]
        attestation = item[1]
        broadcast_list = get_all_sets_for_validator(bignetwork, sender)

        # Adding the attestation to network items
        networkItem = NetworkAttestation(item=attestation, info_sets=broadcast_list)
        bignetwork.attestations.append(networkItem)

        # Update list of validators who received a new item
        for info_set_index in broadcast_list:
            broadcast_validators |= set(bignetwork.sets[info_set_index].validators)

    ask_to_check_backlog(bignetwork, broadcast_validators)
    



def update_network(network: MaliciousNetwork) -> None:
    # The "heartbeat" of the network. When called, items propagate one step further on the network.

    # We need to propagate both blocks and attestations
    item_sets = [network.blocks, network.attestations]

    # These are the validators who receive a new item (block or attestation)
    broadcast_validators = set()

    for item_set in item_sets:
        for item in item_set:
            # For each item, we find the new validators who hear about it for the first time
            # and the validators who already do. Items propagate from validators who know about them.
            known_validators = set()
            for info_set in item.info_sets:
                known_validators = known_validators.union(set(network.sets[info_set].validators))

            # When a validator belongs to a set A where the item was propagated AND
            # to a set B where it wasn't, the validator propagates the item to set B
            unknown_sets = [i for i, s in enumerate(network.sets) if i not in item.info_sets]
            for unknown_set in unknown_sets:
                new_validators = set(network.sets[unknown_set].validators)
                for new_validator in new_validators:
                    if new_validator in known_validators and new_validator:
                        item.info_sets.append(unknown_set)
                        broadcast_validators |= new_validators
                        break

    ask_to_check_backlog(network, broadcast_validators)
