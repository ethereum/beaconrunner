class MaliciousValidator(BRValidator):
    
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
            return private_block_release(self, known_items, malicious_data)
            malicious_data.malicious_head = None
            malicious_data.latest_malicious_slot = None

        # if malicious_data.head is None:
        # # If the attack isn't currently ongoing
        #     return honest_propose(self, known_items)