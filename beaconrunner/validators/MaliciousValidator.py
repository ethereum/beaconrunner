from copy import *

from ..specs import (
    Attestation, SignedBeaconBlock,
    ValidatorIndex,
    SECONDS_PER_SLOT, SLOTS_PER_EPOCH,
)
from ..validatorlib import (
    BRValidator,
    honest_attest, honest_propose, 
)

from ..malicious import (
    MaliciousData,
    private_attest, private_block_release,
)

from .ASAPValidator import (
    ASAPValidator
)

class MaliciousValidator(ASAPValidator):
    
    validator_behaviour = "malicious"
    
    def __init__(self, validator_index: ValidatorIndex):
        super().__init__(validator_index)
        self.attacking = False
    
    def malicious_attest(self, known_items, malicious_data):
        # Not the moment to attest
        if self.data.current_attest_slot != self.data.slot:
            return None

        time_in_slot = (self.store.time - self.store.genesis_time) % SECONDS_PER_SLOT

        # Too early in the slot / didn't receive block
        if not self.data.received_block and time_in_slot < 4:
            return None

        # Already attested for this slot
        if self.data.last_slot_attested == self.data.slot:
            return None
            
        if malicious_data.malicious_validator_indices.count(self.data.current_proposer_indices[self.data.slot % SLOTS_PER_EPOCH]) > 0:
            return private_attest(self, known_items, malicious_data)

        # I don't want to be malicious for now...
        return None
    
    def malicious_propose(self, known_items, malicious_data):
        # When should a malicious validator perform the attack?
        time_in_slot = (self.store.time - self.store.genesis_time) % SECONDS_PER_SLOT

        if not self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            self.attacking = False # Could lead to weird behaviours when the malicious attacker gets to propose twice in a row...
            return None

        if self.data.last_slot_proposed == self.data.slot:
            return None
        
        # Already attacking
        if self.attacking:
            return None
        
        # TODO: Should the attacker always attack whenever it is their turn to propose?
        #       Perhaps the attacker wants to wait until the attestation committees are favorable
        #       (enough wasted honest votes after the next slot)
        #       Maybe the attacker shouldn't attack when the next block is proposed by an attacker too
        if True:
            self.attacking = True
            return private_block_release(self, known_items, malicious_data)
        
        # I don't want to be malicious for now...
        self.attacking = False
        return None
    
    def propose(self, known_items):
        
        if self.attacking:
            return None
        
        return super().propose(known_items)
        