from typing import Optional, Sequence

from model.specs import (
    SignedBeaconBlock, SLOTS_PER_EPOCH,
)

from model.validatorlib import (
    randao_propose,
)

from model.validators.ASAPValidator import ASAPValidator

class RANDAOValidator(ASAPValidator):
    # ytfsitfsiwetwvmhtruhihafqhnxs

    validator_behaviour = "randao"

    def __init__(self, scenario=None, **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario

    def propose(self, known_items) -> Optional[SignedBeaconBlock]:
        """
        Usually behaves like ASAPValidator, but in order to test some randao egde case, may
        propose no block or create a slashable proposing incident.

        Args:
            self (PrudentValidator): Validator
            known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)

        Returns:
            Optional[SignedBeaconBlock]: Either `None` if the validator decides not to propose,
            otherwise a `SignedBeaconBlock` containing attestations
        """

        # Not supposed to propose for current slot
        if not self.data.current_proposer_duties[self.data.slot % SLOTS_PER_EPOCH]:
            return None

        # Already proposed for this slot
        if self.data.last_slot_proposed == self.data.slot:
            return None

        # randao propose
        # scenario types: "honest", "skip", "slashable"
        return randao_propose(self, known_items, self.scenario)
