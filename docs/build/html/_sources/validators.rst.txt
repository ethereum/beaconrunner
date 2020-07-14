validators
==========

Validator behaviours subclass :py:class:`beaconrunner.validatorlib.BRValidator` and use attributes exposed in `self.data` (of type :py:class:`beaconrunner.validatorlib.ValidatorData`) to make decisions.
Subclasses of `BRValidator` must define at least two methods:

- `attest(self, known_items) -> Optional[Attestation]`
- `propose(self, known_items) -> Optional[Attestation]`

.. automodule:: beaconrunner.validators
   :members:
