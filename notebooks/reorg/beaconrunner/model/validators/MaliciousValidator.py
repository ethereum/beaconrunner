from copy import *

from model.specs import (
    config, Attestation, SignedBeaconBlock,
    ValidatorIndex, SLOTS_PER_EPOCH,
    get_beacon_proposer_index, compute_proposer_index,
)
SECONDS_PER_SLOT = config.SECONDS_PER_SLOT

from model.validatorlib import (
    honest_attest, honest_propose,
)

from experiments.templates.reorg.malicious import (
    MaliciousData,
    make_malicious_attestation, make_malicious_block,
)

from model.validators.ASAPValidator import ASAPValidator

class MaliciousValidator(ASAPValidator):

    validator_behaviour = "malicious"
    last_slot_malicious_attested = None

    def __init__(self, validator_index: ValidatorIndex):
        super().__init__(validator_index)
        self.attacking = False

    # TODO
    def malicious_attest(self, known_items, malicious_data, params=None):
        # Not the moment to attest
        if self.data.current_attest_slot != self.data.slot:
            return None

        time_in_slot = (self.store.time - self.store.genesis_time) % SECONDS_PER_SLOT

        # Too early in the slot / didn't receive block
        if not self.data.received_block and time_in_slot < 4:
            return None

        # Already attested for this slot
        if self.last_slot_malicious_attested == self.data.slot:
            return None

        if self.attacking:
            return None

#         if self.last_slot_malicious_attested == self.data.slot:
#             return None
        if malicious_data.malicious_block is None:
            return None

        if self.last_slot_malicious_attested != self.data.slot:
            self.attacking = True
            self.last_slot_malicious_attested = self.data.slot
            #print("{} has produces a malicious attest".format(self.validator_index))
            return make_malicious_attestation(self, known_items, malicious_data)

        # I don't want to be malicious for now...
        self.attacking = False
        return None

    def attest(self, known_items, params=None):

        if self.last_slot_malicious_attested == self.data.slot:
            return None

        return super().attest(known_items, params=params)

    def malicious_propose(self, known_items, malicious_data, params=None):
        # When should a malicious validator perform the attack?
        time_in_slot = (self.store.time - self.store.genesis_time) % SECONDS_PER_SLOT

        if not self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            self.attacking = False
            # TODO: Could lead to weird behaviours when the malicious attacker gets to propose twice in a row...
            return None

        if self.data.last_slot_proposed == self.data.slot:
            return None

        # Already attacking
        if self.attacking:
            return None

        #should_attack_based_on_next_proposer = False

        head_root = self.get_head()
        current_slot = self.data.slot
        required_state = self.process_to_slot(head_root, current_slot + 1)
        next_proposer_index = get_beacon_proposer_index(required_state)



#         if(current_slot == SLOTS_PER_EPOCH) or (malicious_data.percentage_malicious_validators_attesting_forward <= 0):
#             should_attack = False
#         elif next_proposer_index in malicious_data.malicious_validator_indices:
#             should_attack = False
#         else:
#             should_attack = True

            #should_attack = True

        # TODO: Should the attacker always attack whenever it is their turn to propose?
        #       Perhaps the attacker wants to wait until the attestation committees are favorable
        #       (enough wasted honest votes after the next slot)
        #       Maybe the attacker shouldn't attack when the next block is proposed by an attacker too
        if (current_slot != (SLOTS_PER_EPOCH) or 1) and (next_proposer_index not in malicious_data.malicious_validator_indices) and (malicious_data.percentage_malicious_validators_attesting_forward >= 0):
            self.attacking = True
            return make_malicious_block(self, known_items, malicious_data)

        # I don't want to be malicious for now...
        self.attacking = False
        return None

    def propose(self, known_items, params=None):

        if self.attacking:
            return None

        return super().propose(known_items, params=params)
