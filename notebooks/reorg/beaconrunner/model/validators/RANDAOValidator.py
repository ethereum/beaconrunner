from typing import Optional, Sequence

from model.specs import (
    SignedBeaconBlock, SLOTS_PER_EPOCH,
)

from model.validatorlib import (
    honest_propose, slashable_propose
)

from model.validators.ASAPValidator import ASAPValidator

#############################
# Main new function handling different scenarios to inspect randao: randao_propose
#############################

def randao_propose(validator, known_items, scenario="honest"):
    """
    Returns an honest block, using the current LMD-GHOST head and all known, aggregated, attestations, except when it's the second-last slot.
    In the last slot the RANDAOValidator proposes a block dependent on the scenario: Either "honest", "skip" or "slashable" blocks are proposed (or skipped).

    Args:
        validator (BRValidator): The proposing validator
        known_items (Dict): Known blocks and attestations received over-the-wire (but perhaps not included yet in `validator.store`)

    Returns:
        SignedBeaconBlock: The honest proposed block.
    """
    # Check if selected proposer has been slashed previously. If yes, skip proposing a block, since it will be considered invalid by honest validatory anyway!
    # Why? In process_block_header() it is checked that a proposer is not slashed with: `assert not proposer.slashed`
    if validator.data.is_slashed == True:
        print("* {} slashed already; is shutting up!".format(validator.validator_index))
        validator.data.last_slot_proposed = validator.data.slot
        return None

    # Check if current slot is epoch's last slot
    is_secondlast_slot = True if (validator.data.slot + 2) % SLOTS_PER_EPOCH == 0 else False

    if is_secondlast_slot == False or scenario == "honest":
        return honest_propose(validator, known_items)

    elif scenario == "skip": # Skip block at epoch's last slot
        print(validator.validator_index, "skipping block for slot", validator.data.slot)
        # TODO: Ensure that validators knows it has already proposed something at this slot (actively proposed nothing). We get all the print statements when running the simulation, because every time the validator is pinged and then thinks "oh i need to propose and then goes on to propose nothing (again and again...)"
        # QUESTION: below line works to the extent that print statement is only repeated twice now. But why twice?!
        validator.data.last_slot_proposed = validator.data.slot
        return None # Skip proposing a block

    else: # scenario "C": "slashable" proposing event at final slot
        return slashable_propose(validator, known_items)

class RANDAOValidator(ASAPValidator):
    # ytfsitfsiwetwvmhtruhihafqhnxs

    validator_behaviour = "randao"

    def __init__(self, scenario=None, **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario

    def propose(self, known_items, params=None) -> Optional[SignedBeaconBlock]:
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
