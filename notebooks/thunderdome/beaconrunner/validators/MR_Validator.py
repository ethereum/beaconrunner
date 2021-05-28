class MRValidator(BRValidator):

  validator_behaviour = "MR"
  
  def __init__(self, malicious_validators: Sequence[ValidatorIndex]):
    super().__init__()
    self.malicious_validators = malicious_validators
    
  def attest(self, known_items) -> Optional[Attestation]:
        """ 
        Args:
            self (MRValidator): Validator
            known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)
        Returns:
            Optional[Attestation]: Either `None` if the validator decides not to attest,
            otherwise an `Attestation`
        """

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
            
        if self.malicious_validators.count(self.data.current_proposer_indices[self.data.slot % SLOTS_PER_EPOCH]) > 0:
            return dishonest_attest(self, known_items)

        # honest attest
        return honest_attest(self, known_items)

   def propose(self, known_items) -> Optional[SignedBeaconBlock]:


        # Not supposed to propose for current slot
        if not self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            return None

        # Already proposed for this slot
        if self.data.last_slot_proposed == self.data.slot:
            return None

        # honest propose
        return dihonest_propose(self, known_items)

  