import gym
from gym.spaces import Dict, Discrete
import numpy as np
from random import random
from copy import copy

class BackgammonEnv(gym.Env):

    # Declaration and Initialization
    def __init__(self):
  
        # https://en.wikipedia.org/wiki/Unary_coding 
        # http://www.scholarpedia.org/article/User:Gerald_Tesauro/Proposed/Td-gammon
        # https://www.bkgm.com/rgb/rgb.cgi?view+610
   
        low = np.zeros((97, 1))
        high = np.ones((96, 1))

        for i in range(3, 97, 4):
            high[i] = 6.0

        high.insert(0,7.5)

        self.observation_space = Dict(
            {
                'W': Dict(
                    {
                        'board': Box(low=low, high=high,dtype=np.float32),
                        'menoff': Box(low=0,high=1,dtype=np.float32),
                        'turn': Discrete(2)
                    }
                ),
                'B': Dict(
                    {
                        'board': Box(low=low, high=high,dtype=np.float32),
                        'menoff': Box(low=0,high=1,dtype=np.float32),
                        'turn': Discrete(2)
                    }
                )
            }
        )

        self.action_space = None

        self.state = {
            'W': {
                'pos':np.zeros((25, 4)), 
                'barmen': 0,
                'menoff': 0,
                'turn': 0
            },
            'B': {
                'pos': np.zeros((25, 4)),
                'barmen': 0,
                'menoff': 0,
                'turn': 0
            }
        }

        self.starting_pos = np.zeros((24, 4))

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

        self.state = {
            'W': {
                'pos':starting_pos, 
                'barmen': 0,
                'menoff': 0,
                'turn': int(coin)
            },
            'B': {
                'pos':starting_pos.reverse(), # Black's pieces mirror White's
                'barmen': 0,
                'menoff': 0,
                'turn': 1-int(coin)
            }
        }

        observation = copy(state)
        observation['W']['pos'] = observation['W']['pos'].flatten().tolist()
        observation['B']['pos'] = observation['B']['pos'].flatten().tolist()

        return observation

    # Step
    def step(self, action):
        w_win = False
        b_win = False
        w_gammon = False
        b_gammon = False
        # what should actions look like?
        # given a dice roll, we need to create a list of valid actions
        # the action needs to contain: how many pieces will be moved, which pieces will be moved, how far the pieces will be moved
        # from that, step() needs to calculate the results of those actions: moving pieces, blots, removing pieces from the board/bar,...
        # ok so lets assume that an action is a list
        # action = [[position_from_which_to_move_a_piece,how_many_spaces_to_move],[position_from_which_to_move_a_piece,how_many_spaces_to_move]]
        # or
        # action = [[position_from_which_to_move_a_piece,how_many_spaces_to_move],[position_from_which_to_move_a_piece,how_many_spaces_to_move],[position_from_which_to_move_a_piece,how_many_spaces_to_move],[position_from_which_to_move_a_piece,how_many_spaces_to_move]]
        # problem: the bar is not a position per se? And I can't index it the way I'd index a board position because 'bar' is not part of the 'pos' array in the observation
        for move in action:
            pass
        
        # switch turn order
        self.state['W']['turn'] = 1-self.state['W']['turn']
        self.state['B']['turn'] = 1-self.state['W']['turn']

        # check for winning/gammon conditions

        if self.state['W']['menoff'] ==  1 and self.state['B']['pos']:
            w_gammon = True
        elif self.state['W']['menoff'] ==  1:
            w_win = True
        elif 
        elif self.state['B']['menoff'] ==  1:
            b_win = True



        # make observation


        #reward = 2 if w_gammon elif 1 if w_win elif -1 if b_win elif -2 if b_gammon else 0
        if w_gammon:
            reward = 2 
        elif w_win:
            reward = 1
        elif b_win:
            reward = -1
        elif b_gammon:
            -2
        else:
            reward = 0
        
        done = True if w_gammon or w_win or b_win or b_gammon else False

        return observation, reward, done,

    # # Rendering
    # def render(self):
    #     pass

    # def _render_frame(self):
    #     pass

    # # Close
    # def close(self):
    #     pass