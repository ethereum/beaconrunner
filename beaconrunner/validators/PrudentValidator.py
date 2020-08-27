from typing import Optional

from ..specs import (
    Attestation, SignedBeaconBlock,
    SECONDS_PER_SLOT, SLOTS_PER_EPOCH,
)
from ..validatorlib import (
    BRValidator,
    honest_attest, honest_propose
)

class PrudentValidator(BRValidator):
    # I believe in you

    validator_behaviour = "prudent"

    def attest(self, known_items) -> Optional[Attestation]:
        """
        Returns an honest `Attestation` as soon as a block was received for the
        attesting slot *or* at least 8 seconds (`2 * SECONDS_PER_SLOT / 3`) have elapsed.
        Checks whether an attestation was produced for the same slot to avoid slashing.

        Args:
            self (PrudentValidator): Validator
            known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)

        Returns:
            Optional[Attestation]: Either `None` if the validator decides not to attest,
            otherwise an honest `Attestation`
        """

        # Not the moment to attest
        if self.data.current_attest_slot != self.data.slot:
            return None

        time_in_slot = (self.store.time - self.store.genesis_time) % SECONDS_PER_SLOT

        # Too early in the slot / didn't receive block
        if not self.data.received_block and time_in_slot < 8:
            return None

        # Already attested for this slot
        if self.data.last_slot_attested == self.data.slot:
            return None

        # honest attest
        return honest_attest(self, known_items)

    def propose(self, known_items) -> Optional[SignedBeaconBlock]:
        """
        Returns an honest `SignedBeaconBlock` as soon as the slot where
        the validator is supposed to propose starts.
        Checks whether a block was proposed for the same slot to avoid slashing.

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

        # honest propose
        return honest_propose(self, known_items)
