from typing import Optional, Sequence

from model.specs import (
    SignedBeaconBlock, SLOTS_PER_EPOCH,
)

from model.validators.ASAPValidator import ASAPValidator

class LazyValidator(ASAPValidator):
    # I believe in you

    validator_behaviour = "lazy"

    def propose(self, known_items, scenario="honest") -> Optional[SignedBeaconBlock]:
        """
        Never proposes a block! Remember, it's the lazy proposer at work here!

        Args:
            self (PrudentValidator): Validator
            known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)

        Returns:
            `None`
        """

        # Not supposed to propose for current slot
        if not self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            return None

        # Already proposed for this slot
        if self.data.last_slot_proposed == self.data.slot:
            return None

        # I am lazy, I don't want to propose a block!
        print("* {} is feeling lazy in slot {}; not proposing anything!".format(self.validator_index, self.data.slot))
        self.data.last_slot_proposed = self.data.slot
        return None
