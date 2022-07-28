import gym
from gym.spaces import Dict, Discrete
from typing import Optional
from gym.utils.renderer import Renderer
import numpy as np
from random import random

class BackgammonEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # Declaration and Initialization
    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata['render_modes'] 
        self.render_mode = render_mode # currently this does not do anything
        
        # There are 24 locations on the board
        # For each of the two players the number of that player's pieces at a given location ('pos') is encoded with four units
            # "The first three units encoded separately the cases of one man, two men, and three men, while the fourth unit encoded the number of men beyond 3." - *citation*
            # this is called a "truncated unary encoding": https://en.wikipedia.org/wiki/Unary_coding, http://www.scholarpedia.org/article/User:Gerald_Tesauro/Proposed/Td-gammon
            # i.e. n = 0: [0,0,0,0], n = 1: [1,0,0,0], n = 2: [1,1,0,0], n = 3: [1,1,1,0], 3<n<=15: [1,1,1,0.5*(n-3)]
        # That is 96 units each for a subtotal of 192 units
        # Additionally we require a subtotal of 6 units, which consist of 3 units per player to encode the number of that player's pieces on the bar ('barmen'), off the board ('men off'), and whether it is that player's turn or not ('turn)
        # This gives us a total of 198 units which make up an observation
        
        # Each player gets 96 units to encode their number of pieces at a given location on the board
        # Units have a value of either 0 or 1, except every fourth unit, which can have a value of up to 6
        low = np.zeros((96, 1))
        high = np.ones((96, 1))

        # set every fourth value in 'high' to 6.0
        for i in range(3, 97, 4):
            high[i] = 6.0

        # I define the observation space as a dictionary space to make it more readable to humans
        # But Dict observations need to be converted to flat arrays using the 'FlattenObservation' wrapper later
        self.observation_space = Dict(
            {
                'W': Dict(
                    {
                        # might not be correct if we use 24 arrays to group the values which encode each position?
                        'pos': Box(low=low, high=high,dtype=np.float32), # truncated unary encoding explained above, shape is inferred from the shape of 'low' and 'high'
                        'barmen': Box(low=0,high=7.5,dtype=np.float32), # number of pieces on the bar divided by two
                        'menoff': Box(low=0,high=1,dtype=np.float32), # number of pieces off the board, expressed as a fraction of total pieces i.e. n/15
                        'turn': Discrete(2) # 1 if it is White's turn, 0 if not
                    }
                ),
                'B': Dict(
                    {
                        'pos': Box(low=low, high=high,dtype=np.float32),
                        'barmen': Box(low=0,high=7.5,dtype=np.float32),
                        'menoff': Box(low=0,high=1,dtype=np.float32),
                        'turn': Discrete(2) # 1 if it is Black's turn, 0 if not
                    }
                )
            }
        )

        # self.observation_space and self.action_space merely specify the format of valid observations and actions for the benefit of humans. They are not necessary for the functioning of the environment(?)
        # Since the action space is very complicated (it changes depending on current board configuration and the result of the random dice roll ), I make no attempt at specifying it and simply set it to 'None' instead
        self.action_space = None

        # define starting positions for each side
        # note: opponents play by moving their pieces to their respective 'opposite ends' of the board
        # positions 1 and 24 correspond to White's 'start' and 'end' positions respectively
        # vice versa 24 is Black's start, 1 is Black's 'end'

        # first, create an empty board
        # for now, the four values encoding each position are going to get grouped into a list
        self.starting_pos = np.zeros((24, 4))

        # place pieces to match the game's starting positions
        two = [1,1,0,0]
        three = [1,1,1,0]
        five = [1,1,1,1]

        self.starting_pos[0] = five
        self.starting_pos[11] = two
        self.starting_pos[17] = five
        self.starting_pos[19] = three


    # Reset
    def reset(self): 

        # 'coin flip' to determine which side goes first
        coin = random() > 0.5

        observation = {
            'W': {
                'pos':starting_pos.flatten().tolist(), 
                'barmen': 0,
                'menoff': 0,
                'turn': int(coin)
            },
            'B': {
                'pos':starting_pos.reverse().flatten().tolist(), # Black's pieces mirror White's
                'barmen': 0,
                'menoff': 0,
                'turn': 1-int(coin)
            }
        }

        return observation

    # Step
    def step(self, action):
        # each player can moves up to four pieces in one turn. pieces need to be subtracted from their previous location and then added to their new locations. if one or more moves result in
        # one or more of the opponent's piece being moved to the bar, we need to subtract those pieces from the opponent's board and add them to the bar. if pieces are moved off the board, this needs to be considered
        # if a player has to remove a piece from the bar, this has to be considered
        # if a player can't do anything, this has to be considered
        # if nothing else changes, we still need to change whose turn it is

        # check for winning/gammon conditions
        # return reward and end the episode if reached
        return observation, reward, done, info

    # # Rendering
    # def render(self):
    #     pass

    # def _render_frame(self):
    #     pass

    # # Close
    # def close(self):
    #     pass