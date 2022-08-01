import gym
from gym.spaces import Dict, Discrete
import numpy as np
from random import random
from copy import copy

class BackgammonEnv(gym.Env):
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
                'board':np.zeros((24, 4)).insert(0,[0]), 
                'menoff': 0,
                'turn': 0
            },
            'B': {
                'board': np.zeros((24, 4)).insert(0,[0]),
                'menoff': 0,
                'turn': 0
            }
        }

        two = [1,1,0,0]
        three = [1,1,1,0]
        five = [1,1,1,1]

        self.starting_pos = np.zeros((24, 4))

        self.starting_pos[0] = five
        self.starting_pos[11] = two
        self.starting_pos[17] = five
        self.starting_pos[19] = three

        self.starting_pos.insert(0,[0])

    def _get_obs(self):
        observation = copy(state)
        observation['W']['board'] = observation['W']['board'].flatten().tolist()
        observation['B']['board'] = observation['B']['board'].flatten().tolist()
        return observation

    def _get_info(self):
        pass

    def reset(self): 

        coin = random() > 0.5

        # W:    0 1 ... 24
        # B:      24 ... 1 0

        self.state = {
            'W': {
                'board':starting_pos, 
                'menoff': 0,
                'turn': int(coin)
            },
            'B': {
                'board':starting_pos,
                'menoff': 0,
                'turn': 1-int(coin)
            }
        }

        return self._get_obs()

    # translating from one board to the other: n+25-(2n)

    def step(self, action):
        w_win = False
        b_win = False
        w_gammon = False
        b_gammon = False
        
        for move in action:
            pass

        self.state['W']['turn'] = 1-self.state['W']['turn']
        self.state['B']['turn'] = 1-self.state['W']['turn']

        observation = self._get_obs()

        reward = 0

        # White wins, Black is gammoned
        if self.state['W']['menoff'] ==  1 and self.state['B']['menoff'] = 0:
            reward = 2
        # White wins
        elif self.state['W']['menoff'] ==  1:
            reward = 1
        # Black wins, White is gammoned
        elif self.state['B']['menoff'] ==  1 and self.state['W']['menoff'] = 0: 
            reward = -2
        # Black wins
        elif self.state['B']['menoff'] ==  1:
            reward = -1
        
        done = reward != 0

        return observation, reward, done