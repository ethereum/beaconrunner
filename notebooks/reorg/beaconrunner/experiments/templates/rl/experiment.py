import copy
import numpy as np

from radcad import Simulation, Model

import matplotlib.pyplot as plt 
#from experiments.visualizations.plot_chain import draw_graph

from model.system_parameters import parameters
from model.state_update_blocks import state_update_blocks
import model.specs as specs
import model.simulator as simulator
from model.state_variables import create_default_initial_network

from model.validators.ASAPValidator import ASAPValidator
from model.validators.RLValidator import RLValidator

num_validators = 12
validators = (
    [ASAPValidator(validator_index=i) for i in range(num_validators-1)] +
    [RLValidator(validator_index=(num_validators-1))]
)

import gym
from gym import spaces

class BeaconEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BeaconEnv, self).__init__()
        
                # Reset to initial conditions
        network = create_default_initial_network(validators=validators)
        self.model = Model(
            params=parameters,
            initial_state={
                "network": network
            },
            state_update_blocks=state_update_blocks,
        )
        self.model.params.update({
            "num_epochs": [4],
            "rl_actions": [[]]
        })
        self.model._deepcopy = False

        self.length = simulator.get_timesteps(Simulation(model=self.model)) / 12

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([12]))
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([12]))
        self.balance = 32e9 # Used to keep track of balance differentials
        self.network = None
        

    def reset(self):
        # Reset to initial conditions
        network = create_default_initial_network(validators=validators)
        self.model = Model(
            params=parameters,
            initial_state={
                "network": network
            },
            state_update_blocks=state_update_blocks,
        )
        self.model.params.update({
            "num_epochs": [4],
            "rl_actions": [[]]
        })
        self.model._deepcopy = False

        self.length = simulator.get_timesteps(Simulation(model=self.model)) / 12

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([12]))
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([12]))
        self.balance = 32e9 # Used to keep track of balance differentials
        self.network = None
        
        return np.array([0])

    def step(self, action):
        # Execute one time step within the environment
        self.model.params.update({
            "rl_actions": [self.model.params["rl_actions"][0] + [action[0]]]
        })
        model_generator = iter(self.model)
        for i in range(specs.config.SECONDS_PER_SLOT):
            self.model = next(model_generator)

        network = self.model.state["network"]
        validators = network.validators
        rl_validator = validators[-1]
        head = specs.get_head(rl_validator.store)
        current_state = rl_validator.store.block_states[head].copy()
        
        current_balance = int(current_state.balances[num_validators-1])
        reward = current_balance - self.balance
        self.balance = current_balance
        
        obs = np.array([action[0]])
        info = {}

        self.length -= 1
        done = self.length <= 0
        
        self.network = network

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        # if self.network == None:
        #     pass
        # else:
        #     plot_chain_tree(self.network)
        pass

beacon_env = BeaconEnv()
