import copy

from experiments.default_experiment import experiment

import model.parts.observers as observers
import experiments.templates.randao.observers as randao_observers

import model.simulator as simulator
from model.network import Network, NetworkSet
from model.validators.RANDAOValidator import RANDAOValidator

# Make a copy of the default experiment to avoid mutation
experiment = copy.deepcopy(experiment)

scenarios = ["honest", "skip", "slashable"]
# scenarios = ["slashable"]

randao_observers = {
    "current_epoch": randao_observers.current_epoch,
    "current_slot": randao_observers.current_slot,
    "current_epoch_proposer_indices": randao_observers.get_current_proposer_indices,
    "next_epoch_proposer_indices": randao_observers.get_next_proposer_indices,
    "plus_2_epoch_proposer_indices": randao_observers.get_plus_2_epoch_proposer_indices,
    "balances": randao_observers.balances,
}

def create_initial_network(scenario):
    num_validators = 20

    set_a = NetworkSet(validators=list(range(num_validators)))
    network_sets = list([set_a])

    validators = [RANDAOValidator(validator_index=i, scenario=scenario) for i in range(num_validators)]

    # Create a genesis state
    (genesis_state, genesis_block) = simulator.get_genesis_state_block(validators, seed="let's play randao")

    # Validators load the state
    [v.load_state(genesis_state.copy(), genesis_block.copy()) for v in validators]

    # We skip the genesis block
    simulator.skip_genesis_block(validators)

    network = Network(validators=validators, sets=network_sets)
    return network

def create_scenario_model(scenario):
    new_model = copy.deepcopy(experiment.simulations[0].model)
    new_model.initial_state["network"] = create_initial_network(scenario)
    parameter_overrides = {
        "num_epochs": [5]
    }
    new_model.params.update(parameter_overrides)

    (observed_ic, observed_psubs) = observers.add_observers(
        new_model.initial_state,
        new_model.state_update_blocks,
        randao_observers
    )

    new_model.initial_state = observed_ic
    new_model.state_update_blocks = observed_psubs

    return new_model

def create_simulation(scenario):
    new_simulation = copy.deepcopy(experiment.simulations[0])
    new_simulation.model = create_scenario_model(scenario)
    new_simulation.timesteps = simulator.get_timesteps(new_simulation)
    return new_simulation

experiment.simulations = [create_simulation(scenario) for scenario in scenarios]
