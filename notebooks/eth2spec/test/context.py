import pytest

from eth2spec.phase0 import spec as spec_phase0
from eth2spec.altair import spec as spec_altair
from eth2spec.utils import bls

from .exceptions import SkippedTest
from .helpers.genesis import create_genesis_state
from .utils import vector_test, with_meta_tags

from random import Random
from typing import Any, Callable, NewType, Sequence, TypedDict, Protocol

from lru import LRU

from importlib import reload


def reload_specs():
    reload(spec_phase0)
    reload(spec_altair)


# Some of the Spec module functionality is exposed here to deal with phase-specific changes.

SpecForkName = NewType("SpecForkName", str)
ConfigName = NewType("ConfigName", str)

PHASE0 = SpecForkName('phase0')
ALTAIR = SpecForkName('altair')

# Experimental phases (not included in default "ALL_PHASES"):
MERGE = SpecForkName('merge')
SHARDING = SpecForkName('sharding')
CUSTODY_GAME = SpecForkName('custody_game')
DAS = SpecForkName('das')

ALL_PHASES = (PHASE0, ALTAIR)

MAINNET = ConfigName('mainnet')
MINIMAL = ConfigName('minimal')

ALL_CONFIGS = (MINIMAL, MAINNET)

# The forks that output to the test vectors.
TESTGEN_FORKS = (PHASE0, ALTAIR)

# TODO: currently phases are defined as python modules.
# It would be better if they would be more well-defined interfaces for stronger typing.


class Spec(Protocol):
    version: str


class SpecPhase0(Spec):
    ...


class SpecAltair(Spec):
    ...


class SpecForks(TypedDict, total=False):
    PHASE0: SpecPhase0
    ALTAIR: SpecAltair


def _prepare_state(balances_fn: Callable[[Any], Sequence[int]], threshold_fn: Callable[[Any], int],
                   spec: Spec, phases: SpecForks):

    p0 = phases[PHASE0]
    balances = balances_fn(p0)
    activation_threshold = threshold_fn(p0)

    state = create_genesis_state(spec=p0, validator_balances=balances,
                                 activation_threshold=activation_threshold)
    # TODO: upgrade to merge spec, and later sharding.
    if spec.fork == ALTAIR:
        state = phases[ALTAIR].upgrade_to_altair(state)

    return state


_custom_state_cache_dict = LRU(size=10)


def with_custom_state(balances_fn: Callable[[Any], Sequence[int]],
                      threshold_fn: Callable[[Any], int]):
    def deco(fn):

        def entry(*args, spec: Spec, phases: SpecForks, **kw):
            # make a key for the state
            key = (spec.fork, spec.CONFIG_NAME, spec.__file__, balances_fn, threshold_fn)
            global _custom_state_cache_dict
            if key not in _custom_state_cache_dict:
                state = _prepare_state(balances_fn, threshold_fn, spec, phases)
                _custom_state_cache_dict[key] = state.get_backing()

            # Take an entry out of the LRU.
            # No copy is necessary, as we wrap the immutable backing with a new view.
            state = spec.BeaconState(backing=_custom_state_cache_dict[key])
            kw['state'] = state
            return fn(*args, spec=spec, phases=phases, **kw)
        return entry
    return deco


def default_activation_threshold(spec):
    """
    Helper method to use the default balance activation threshold for state creation for tests.
    Usage: `@with_custom_state(threshold_fn=default_activation_threshold, ...)`
    """
    return spec.MAX_EFFECTIVE_BALANCE


def zero_activation_threshold(spec):
    """
    Helper method to use 0 gwei as the activation threshold for state creation for tests.
    Usage: `@with_custom_state(threshold_fn=zero_activation_threshold, ...)`
    """
    return 0


def default_balances(spec):
    """
    Helper method to create a series of default balances.
    Usage: `@with_custom_state(balances_fn=default_balances, ...)`
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


with_state = with_custom_state(default_balances, default_activation_threshold)


def low_balances(spec):
    """
    Helper method to create a series of low balances.
    Usage: `@with_custom_state(balances_fn=low_balances, ...)`
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8
    # Technically the balances cannot be this low starting from genesis, but it is useful for testing
    low_balance = 18 * 10 ** 9
    return [low_balance] * num_validators


def misc_balances(spec):
    """
    Helper method to create a series of balances that includes some misc. balances.
    Usage: `@with_custom_state(balances_fn=misc_balances, ...)`
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8
    balances = [spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances


def low_single_balance(spec):
    """
    Helper method to create a single of balance of 1 Gwei.
    Usage: `@with_custom_state(balances_fn=low_single_balance, ...)`
    """
    return [1]


def large_validator_set(spec):
    """
    Helper method to create a large series of default balances.
    Usage: `@with_custom_state(balances_fn=default_balances, ...)`
    """
    num_validators = 2 * spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


def single_phase(fn):
    """
    Decorator that filters out the phases data.
    most state tests only focus on behavior of a single phase (the "spec").
    This decorator is applied as part of spec_state_test(fn).
    """
    def entry(*args, **kw):
        if 'phases' in kw:
            kw.pop('phases')
        return fn(*args, **kw)
    return entry


# BLS is turned on by default, it can be disabled in tests by overriding this, or using `--disable-bls`.
# *This is for performance purposes during TESTING, DO NOT DISABLE IN PRODUCTION*.
# The runner of the test can indicate the preferred setting (test generators prefer BLS to be ON).
# - Some tests are marked as BLS-requiring, and ignore this setting.
#    (tests that express differences caused by BLS, e.g. invalid signatures being rejected)
# - Some other tests are marked as BLS-ignoring, and ignore this setting.
#    (tests that are heavily performance impacted / require unsigned state transitions)
# - Most tests respect the BLS setting.
DEFAULT_BLS_ACTIVE = True


is_pytest = True


def dump_skipping_message(reason: str) -> None:
    message = f"[Skipped test] {reason}"
    if is_pytest:
        pytest.skip(message)
    else:
        raise SkippedTest(message)


def spec_test(fn):
    # Bls switch must be wrapped by vector_test,
    # to fully go through the yielded bls switch data, before setting back the BLS setting.
    # A test may apply BLS overrides such as @always_bls,
    #  but if it yields data (n.b. @always_bls yields the bls setting), it should be wrapped by this decorator.
    #  This is why @alway_bls has its own bls switch, since the override is beyond the reach of the outer switch.
    return vector_test()(bls_switch(fn))


# shorthand for decorating @spectest() @with_state @single_phase
def spec_state_test(fn):
    return spec_test(with_state(single_phase(fn)))


def expect_assertion_error(fn):
    bad = False
    try:
        fn()
        bad = True
    except AssertionError:
        pass
    except IndexError:
        # Index errors are special; the spec is not explicit on bound checking, an IndexError is like a failed assert.
        pass
    if bad:
        raise AssertionError('expected an assertion error, but got none.')


def never_bls(fn):
    """
    Decorator to apply on ``bls_switch`` decorator to force BLS de-activation. Useful to mark tests as BLS-ignorant.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """
    def entry(*args, **kw):
        # override bls setting
        kw['bls_active'] = False
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 2})(entry)


def always_bls(fn):
    """
    Decorator to apply on ``bls_switch`` decorator to force BLS activation. Useful to mark tests as BLS-dependent.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """
    def entry(*args, **kw):
        # override bls setting
        kw['bls_active'] = True
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 1})(entry)


def bls_switch(fn):
    """
    Decorator to make a function execute with BLS ON, or BLS off.
    Based on an optional bool argument ``bls_active``, passed to the function at runtime.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """
    def entry(*args, **kw):
        old_state = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry


def disable_process_reveal_deadlines(fn):
    """
    Decorator to make a function execute with `process_reveal_deadlines` OFF.
    This is for testing long-range epochs transition without considering the reveal-deadline slashing effect.
    """
    def entry(*args, spec: Spec, **kw):
        if hasattr(spec, 'process_reveal_deadlines'):
            old_state = spec.process_reveal_deadlines
            spec.process_reveal_deadlines = lambda state: None

        yield from fn(*args, spec=spec, **kw)

        if hasattr(spec, 'process_reveal_deadlines'):
            spec.process_reveal_deadlines = old_state

    return with_meta_tags({'reveal_deadlines_setting': 1})(entry)


def with_all_phases(fn):
    """
    A decorator for running a test with every phase
    """
    return with_phases(ALL_PHASES)(fn)


def with_all_phases_except(exclusion_phases):
    """
    A decorator factory for running a tests with every phase except the ones listed
    """
    def decorator(fn):
        return with_phases([phase for phase in ALL_PHASES if phase not in exclusion_phases])(fn)
    return decorator


def with_phases(phases, other_phases=None):
    """
    Decorator factory that returns a decorator that runs a test for the appropriate phases.
    Additional phases that do not initially run, but are made available through the test, are optional.
    """
    def decorator(fn):
        def wrapper(*args, **kw):
            run_phases = phases

            # limit phases if one explicitly specified
            if 'phase' in kw:
                phase = kw.pop('phase')
                if phase not in phases:
                    dump_skipping_message(f"doesn't support this fork: {phase}")
                    return None
                run_phases = [phase]

            if PHASE0 not in run_phases and ALTAIR not in run_phases:
                dump_skipping_message("none of the recognized phases are executable, skipping test.")
                return None

            available_phases = set(run_phases)
            if other_phases is not None:
                available_phases |= set(other_phases)

            # TODO: test state is dependent on phase0 but is immediately transitioned to later phases.
            #  A new state-creation helper for later phases may be in place, and then tests can run without phase0
            available_phases.add(PHASE0)

            # Populate all phases for multi-phase tests
            phase_dir = {}
            if PHASE0 in available_phases:
                phase_dir[PHASE0] = spec_phase0
            if ALTAIR in available_phases:
                phase_dir[ALTAIR] = spec_altair

            # return is ignored whenever multiple phases are ran.
            # This return is for test generators to emit python generators (yielding test vector outputs)
            if PHASE0 in run_phases:
                ret = fn(spec=spec_phase0, phases=phase_dir, *args, **kw)
            if ALTAIR in run_phases:
                ret = fn(spec=spec_altair, phases=phase_dir, *args, **kw)

            # TODO: merge, sharding, custody_game and das are not executable yet.
            #  Tests that specify these features will not run, and get ignored for these specific phases.
            return ret
        return wrapper
    return decorator


def with_configs(configs, reason=None):
    def decorator(fn):
        def wrapper(*args, spec: Spec, **kw):
            available_configs = set(configs)
            if spec.CONFIG_NAME not in available_configs:
                message = f"doesn't support this config: {spec.CONFIG_NAME}."
                if reason is not None:
                    message = f"{message} Reason: {reason}"
                dump_skipping_message(message)
                return None

            return fn(*args, spec=spec, **kw)
        return wrapper
    return decorator


def is_post_altair(spec):
    # TODO: everything runs in parallel to Altair.
    #  After features are rebased on the Altair fork, this can be reduced to just PHASE0.
    if spec.fork in [PHASE0, MERGE, SHARDING, CUSTODY_GAME, DAS]:
        return False
    return True
