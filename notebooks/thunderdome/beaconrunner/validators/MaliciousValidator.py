from copy import *

class MaliciousValidator(BRValidator):
    
    validator_behaviour = "malicious"

    def malicious_propose(self, known_items, malicious_data):
    # When should a malicious validator perform the attack?
        time_in_slot = (self.store.time - self.store.genesis_time) % SECONDS_PER_SLOT

        if not self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            return None

        if self.data.last_slot_proposed == self.data.slot:
            return None

        if self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            malicious_data.malicious_head = self.get_head()
            malicious_data.latest_malicious_slot = self.data.slot
            
        if (malicious_data.malicious_head != None) and (self.data.slot == malicious_data.latest_malicious_slot + 1) and (time_in_slot > 4):
            malicious_data_copy = deepcopy(malicious_data)
            malicious_data.malicious_head = None
            malicious_data.latest_malicious_slot = None
            return private_block_release(self, known_items, malicious_data_copy)
            

        # if malicious_data.head is None:
        # # If the attack isn't currently ongoing
        #     return honest_propose(self, known_items)

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
            
            if malicious_data.malicious_validators.count(self.data.current_proposer_indices[self.data.slot % SLOTS_PER_EPOCH]) > 0:
                malicious_data.malicious_attestations.append(private_attest(self,known_items,malicious_data))
                return None

        # honest attest
            return honest_attest(self, known_items)