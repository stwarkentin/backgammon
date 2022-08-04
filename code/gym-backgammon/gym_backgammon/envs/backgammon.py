import gym
from gym.spaces import Dict, Discrete
import numpy as np
from random import random
from copy import copy

class BackgammonEnv(gym.Env):
    def __init__(self):
  
        # Tesauro's backgammon algorithms encode the board using a specific 'truncated unary encoding' with some quirks. You can read more about it here:
        # https://en.wikipedia.org/wiki/Unary_coding 
        # http://www.scholarpedia.org/article/User:Gerald_Tesauro/Proposed/Td-gammon
        # https://www.bkgm.com/rgb/rgb.cgi?view+610

        # although its purported benefits are not immediately apparent to us (perhaps we've been spoiled by the advances in computational power which have made such optimizations optional?),
        # our encoding is largely based on the one used in Tesauro's 'pubeval.c', with the only changes made being the order of the values, and perhaps the use of two such figurative boards 
        # (one for each player) rather than one 

        # defining some ranges for our observation space
        low = np.zeros((97, 1)) # 1 value for the bar + 4 per each of the 24 positions on the board
        high = np.ones((96, 1)) # the first three of the four values encoding the number of checkers at a given position are either 0 or 1...

        for i in range(3, 97, 4): # but every fourth value can go as high as (15-3)/2 = 6...
            high[i] = 6.0

        high.insert(0,7.5) # and the very first value, which encodes the number of checkers on the bar, can go as high as 15/2 = 7.5

        # to make them more readable to humans, our observations are dictionaries
        # but we can't feed dictionaries to our ANN, so we are going to have to use gym's 'FlattenObservation' wrapper to flatten the dictionary into a single array later
        self.observation_space = Dict(
            {
                # player 'White'
                'W': Dict(
                    {
                        'board': Box(low=low, high=high,dtype=np.float32), # the board, consisting of the bar and 24 'points'
                        'menoff': Box(low=0,high=1,dtype=np.float32), # number of checkers removed from the board as a fraction of the total number of checkers i.e. n/15
                        'turn': Discrete(2) # '1' if it is this player's turn, '0' if not
                    }
                ),
                # player 'Black'
                'B': Dict(
                    {
                        'board': Box(low=low, high=high,dtype=np.float32),
                        'menoff': Box(low=0,high=1,dtype=np.float32),
                        'turn': Discrete(2)
                    }
                )
            }
        )
        
        # which actions can be taken at any given time step depend on the state of the board and the roll of the dice. hence we choose to not attempt to define the action space
        self.action_space = None
        
        # create an empty board:
        # to allow for indexing of board positions, the environment's state uses an array of arrays to store the values of the bar and each point
        # later we simply flatten this array to create observations
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

        # the previously mentioned truncated unary encoding:
        self._encoding = {
            0: np.array([0,0,0,0]),
            1: np.array([1,0,0,0]),
            2: np.array([1,1,0,0]),
            3: np.array([1,1,1,0]),
            4: np.array([1,1,1,0.5]),
            5: np.array([1,1,1,1]),
            6: np.array([1,1,1,1.5]),
            7: np.array([1,1,1,2.0]),
            8: np.array([1,1,1,2.5]),
            9: np.array([1,1,1,3.0]),
            10: np.array([1,1,1,3.5]),
            11: np.array([1,1,1,4.0]),
            12: np.array([1,1,1,4.5]),
            13: np.array([1,1,1,5.0]),
            14: np.array([1,1,1,5.5]),
            15: np.array([1,1,1,6.0]),
        }

        # define the game's starting position:
  
        # create an empty board without a bar
        self.starting_pos = np.zeros((24, 4))

        # place the correct number of checkers in the correct positions
        self.starting_pos[0] = self._encoding[5]
        self.starting_pos[11] = self._encoding[2]
        self.starting_pos[17] = self._encoding[5]
        self.starting_pos[19] = self._encoding[3]

        # add the bar
        self.starting_pos.insert(0,[0])

    def _get_obs(self):
        observation = copy(state)
        observation['W']['board'] = observation['W']['board'].flatten().tolist()
        observation['B']['board'] = observation['B']['board'].flatten().tolist()
        return observation

    def _get_info(self):
        pass

    def reset(self): 

        # a 'coin flip' to determine which side goes first
        coin = random() > 0.5

        # reset the board to the game's starting position and assign a turn order based on the coin flip
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

    def step(self, action):

        # who's turn is it?
        if self.state['W']['turn'] == 1:
            player = 'W'
            opponent = 'B'
        else:
            player = 'B'
            opponent = 'W'

        # assume for now that an action is a list of up to four 'old position - new position' tupels
        # i.e. "remove one checker at this position, add one checker at that position"
        # then, check for blots (translating from one board to the other: n+25-(2n))
        # if there are blots, move the opponents piece to the bar
        for move in action:
            old_pos, new_pos = move
            # get the current number of checkers at the position from which we need to remove a checker
            encoded_checkers = self.state[player][board][old_pos]
            # decode
            for key, value in self._encoding.items():
                if encoded_checkers == value:
                    n_checkers = key
            # subtract a checker
            self.state[player][board][old_pos] = self._encoding[n_checkers-1]


            # get the current number of checkers at the position to which we need to add a checker
            encoded_checkers = self.state.[player][board][new_pos]
            # decode
            for key, value in self._encoding.items():
                if encoded_checkers == value:
                    n_checkers = key
            # add a checker
            self.state[player][board][new_pos] = self._encoding[n_checkers+1]
            
        # moving pieces off the bar

        # bearing off

        # swap whose turn it is
        self.state['W']['turn'] = 1-self.state['W']['turn']
        self.state['B']['turn'] = 1-self.state['W']['turn']

        # get observation
        observation = self._get_obs()

        # reward is zero unless one of four conditions is met
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

        # if one of the four conditions is met, the game is finished and the episode ends
        done = reward != 0

        return observation, reward, done

        
 # W:    0 1 ... 24
# B:      24 ... 1 0

# translating from one board to the other: n+25-(2n)

# idea: replace W and B with 1 and 0 