from typing import Optional

from ..specs import (
    Attestation, SignedBeaconBlock,
)
from ..validatorlib import (
    BRValidator
)

class OfflineValidator(BRValidator):
    # 404 not found

    validator_behaviour = "offline"

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

        # Never attest
        return None

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

        # Never propose
        return None
