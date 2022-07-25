import gym
from gym.spaces import Dict, Discrete
from typing import Optional
from gym.utils.renderer import Renderer
import numpy as np
from random import random

class BackgammonEnv(gym.Env):
    metadata = {'render_modes': ['placeholder'], 'render_fps': 4}

    # Declaration and Initialization
    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata['render_modes'] 
        self.render_mode = render_mode 
        
        #self.window_size = 512
        
        # There are 24 locations on the board
        # For each of the two players the number of that player's pieces at a given location ('boardmen') is encoded with four units
            # "The first three units encoded separately the cases of one man, two men, and three men, while the fourth unit encoded the number of men beyond 3." - *citation*
            # this is called a "truncated unary encoding": https://en.wikipedia.org/wiki/Unary_coding, http://www.scholarpedia.org/article/User:Gerald_Tesauro/Proposed/Td-gammon
            # i.e. 0: [0,0,0,0], 1: [1,0,0,0], 2: [1,1,0,0], 3: [1,1,1,0], 3<n<=15: [1,1,1,0.5*(n-3)]
        # That is 96 units each for a subtotal of 192 units
        # Additionally we require a subtotal of 6 units, which consist of 3 units per player to encode the number of that player's pieces on the bar ('barmen'), off the board ('men off'), and whether it is that player's turn or not ('turn)
        # This gives us a total of 198 units which make up an observation
        # I define the observation space as a dictionary space to make it more readable to humans
        # But Dict observations need to be converted to flat arrays using the 'FlattenObservation' wrapper later
        
        # Each player gets 96 units to encode their number of pieces at a given location on the board
        # Units have a value of either 0 or 1, except every fourth unit, which can have a value of up to 6
        low = np.zeros((96, 1))
        high = np.ones((96, 1))

        # set every fourth value in 'high' to 6.0
        for i in range(3, 97, 4):
            high[i] = 6.0

        self.observation_space = Dict(
            {
                'White': Dict(
                    {
                        'W boardmen': Box(low=low, high=high,dtype=np.float32), # truncated unary encoding explained above, shape is inferred from the shape of 'low' and 'high'
                        'W barmen': Box(low=0,high=1,dtype=np.float32), # number of pieces on the bar, expressed as a fraction of total pieces i.e. n/15
                        'W men off': Box(low=0,high=1,dtype=np.float32), # number of pieces off the board, expressed as a fraction of total pieces i.e. n/15
                        'W turn': Discrete(2) # 1 if it is White's turn, 0 if not
                    }
                ),
                'Black': Dict(
                    {
                        'B boardmen': Box(low=low, high=high,dtype=np.float32),
                        'B barmen': Box(low=0,high=1,dtype=np.float32),
                        'B men off': Box(low=0,high=1,dtype=np.float32),
                        'B turn': Discrete(2) # 1 if it is Black's turn, 0 if not
                    }
                )
            }
        )

        # self.observation_space and self.action_space merely specify the format of valid observations and actions for the benefit of humans. They are not necessary for the functioning of the environment(?)
        # Since the action space is very complicated (it changes depending on current board configuration and the result of the random dice roll ), I make no attempt at specifying it and simply set it to 'None' instead
        self.action_space = None

    # Reset
    def reset(self): 
        # 'coin flip' to determine which player goes first
        # if random() > 0.5:
        #     w_turn = 1
        #     b_turn = 0
        # else:
        #     w_turn = 0
        #     b_turn = 1
        # pass 

        # observation = {
        #     'White'
        # }

    # Step
    def step(self, action):
        pass

    # Rendering
    def render(self):
        pass

    def _render_frame(self):
        pass

    # Close
    def close(self):
        pass
 