import gym
from gym.spaces import Dict, Box, Discrete
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
        low = np.zeros((97,)) # 1 value for the bar + 4 per each of the 24 positions on the board
        high = np.ones((96,)) # the first three of the four values encoding the number of checkers at a given position are either 0 or 1...

        for i in range(3, 97, 4): # but every fourth value can go as high as (15-3)/2 = 6...
            high[i] = 6.0

        high = np.insert(arr=high,obj=0,values=7.5) # and the very first value, which encodes the number of checkers on the bar, can go as high as 15/2 = 7.5
        # to make them more readable to humans, our observations are dictionaries
        # but we can't feed dictionaries to our ANN, so we are going to have to use gym's 'FlattenObservation' wrapper to flatten the dictionary into a single array later
        self.observation_space = Dict(
            {
                # player 'White'
                'W': Dict(
                    {
                        'board': Box(low=low, high=high,dtype=np.float32), # the board, consisting of the bar and 24 'points'
                        'menoff': Box(low=0.0,high=1.0,shape=(1,),dtype=np.float32), # number of checkers removed from the board as a fraction of the total number of checkers i.e. n/15
                        'turn': Discrete(2) # '1' if it is this player's turn, '0' if not
                    }
                ),
                # player 'Black'
                'B': Dict(
                    {
                        'board': Box(low=low, high=high,dtype=np.float32),
                        'menoff': Box(low=0.0,high=1.0,shape=(1,),dtype=np.float32),
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
                'board': np.zeros((24, 4)),
                'barmen': 0,
                'menoff': 0,
                'turn': 0
            },
            'B': {
                'board': np.zeros((24, 4)),
                'barmen': 0,
                'menoff': 0,
                'turn': 0
            }
        }

        # the previously mentioned truncated unary encoding:
        self.encoding = {
            0: np.array([0.,0.,0.,0.]),
            1: np.array([1.,0.,0.,0.]),
            2: np.array([1.,1.,0.,0.]),
            3: np.array([1.,1.,1.,0.]),
            4: np.array([1.,1.,1.,0.5]),
            5: np.array([1.,1.,1.,1.]),
            6: np.array([1.,1.,1.,1.5]),
            7: np.array([1.,1.,1.,2.0]),
            8: np.array([1.,1.,1.,2.5]),
            9: np.array([1.,1.,1.,3.0]),
            10: np.array([1.,1.,1.,3.5]),
            11: np.array([1.,1.,1.,4.0]),
            12: np.array([1.,1.,1.,4.5]),
            13: np.array([1.,1.,1.,5.0]),
            14: np.array([1.,1.,1.,5.5]),
            15: np.array([1.,1.,1.,6.0])
        }

        # define the game's starting position:
  
        # create an empty board without a bar
        self.starting_pos = np.zeros((24, 4))

        # !!! NEW: values where the wrong way around
        # place the correct number of checkers in the correct positions
        self.starting_pos[0] = copy(self.encoding[2])
        self.starting_pos[11] = copy(self.encoding[5])
        self.starting_pos[16] = copy(self.encoding[3])
        self.starting_pos[18] = copy(self.encoding[5])

    # !!! NEW: now parametarized
    def _get_obs(self, obs):
        w_board = obs['W']['board']
        w_board = w_board.flatten()
        
        b_board = obs['B']['board']
        b_board = b_board.flatten()

        observation = []
        observation = np.append(observation,w_board)
        observation = np.append(observation,obs['W']['barmen'])
        observation = np.append(observation,obs['W']['menoff'])
        observation = np.append(observation,obs['W']['turn'])
        observation = np.append(observation,b_board)
        observation = np.append(observation,obs['B']['barmen'])
        observation = np.append(observation,obs['B']['menoff'])
        observation = np.append(observation,obs['B']['turn'])
        return observation

    def _get_info(self):
        pass

    # !!! NEW: return unflattened state
    def reset(self): 

        # a 'coin flip' to determine which side goes first
        coin = int(random()>0.5)

        # reset the board to the game's starting position and assign a turn order based on the coin flip
        self.state = {
            'W': {
                'board':copy(self.starting_pos), 
                'barmen': 0,
                'menoff': 0,
                'turn': coin
            },
            'B': {
                'board':copy(self.starting_pos),
                'barmen': 0,
                'menoff': 0,
                'turn': 1-coin
            }
        }

        return self.state

    # !!! NEW: Indexing, returns unflattened state
    def step(self, action):

        # who's turn is it?
        if self.state['W']['turn'] == 1:
            player = 'W'
            opponent = 'B'
        else:
            player = 'B'
            opponent = 'W'

        # assume for now that an action is a list of up to four 'old position - new position' tupels
        # move = (int,int)
        # action  = [move,move]
        for move in action:

            # 'LIFTING' A CHECKER

            old_pos, new_pos = move
            # are we moving a piece off the bar?
            if old_pos == -1:
                # remove a checker from the bar
                self.state[player]['barmen'] -= 0.5

            else:
                # get the current number of checkers at the position from which we need to remove a checker
                encoded_checkers = self.state[player]['board'][old_pos]
                # decode
                for key, value in self.encoding.items():
                    if np.array_equal(encoded_checkers,value):
                        n_checkers = key
                # subtract a checker
                self.state[player]['board'][old_pos] = copy(self.encoding[n_checkers-1])

            # 'PLACING DOWN' A CHECKER

            # are we bearing off?
            if new_pos == 24:
                self.state[player]['menoff'] += 1/15

            else:
                # get the current number of checkers at the position to which we need to add a checker
                encoded_checkers = self.state[player]['board'][new_pos]
                # decode
                for key, value in self.encoding.items():
                    if np.array_equal(encoded_checkers,value):
                        n_checkers = key
                # add a checker
                self.state[player]['board'][new_pos] = copy(self.encoding[n_checkers+1])

                # check for blots
                mirror_pos = new_pos+23-2*new_pos
                if not np.array_equal(self.state[opponent]['board'][mirror_pos],[0,0,0,0]):
                    # if there is a blot, move the opponent's piece to the bar
                    self.state[opponent]['board'][mirror_pos] = [0,0,0,0]
                    self.state[opponent]['barmen'] += 0.5

            # 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
            #  24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1 0

        # swap whose turn it is
        self.state['W']['turn'] = 1-self.state['W']['turn']
        self.state['B']['turn'] = 1-self.state['B']['turn']

        # !!! nicht mehr hier flatten sondern stadtdessen n der neuen Funktion !!!
        # observation = self._get_obs()

        # reward is zero unless one of four conditions is met:
        reward = 0

        # 1) White wins, Black is gammoned
        if self.state['W']['menoff'] > 0.9 and self.state['B']['menoff'] == 0:
            reward = 2
        # 2) White wins
        elif self.state['W']['menoff'] > 0.9:
            reward = 1
        # 3) Black wins, White is gammoned
        elif self.state['B']['menoff'] > 0.9 and self.state['W']['menoff'] == 0: 
            reward = -2
        # 4) Black wins
        elif self.state['B']['menoff'] > 0.9:
            reward = -1

        # if one of the four conditions is met, the game is finished and the episode ends
        done = reward != 0

        return self.state, reward, done
    
    # !!! NEW: paremetarized copy of stepfunction without reward and done return and flattened
    def get_state(self, observation, action):
        
        # who's turn is it?
        if observation['W']['turn'] == 1:
            player = 'W'
            opponent = 'B'
        else:
            player = 'B'
            opponent = 'W'

        # assume for now that an action is a list of up to four 'old position - new position' tupels
        # move = (int,int)
        # action  = [move,move]
        for move in action:

            # 'LIFTING' A CHECKER

            old_pos, new_pos = move
            
            # are we moving a piece off the bar?
            if old_pos == -1:
                # remove a checker from the bar
                observation[player]['barmen'] -= 0.5

            else:
                # get the current number of checkers at the position from which we need to remove a checker
                encoded_checkers = observation[player]['board'][old_pos]
                # decode
                for key, value in self.encoding.items():
                    if np.array_equal(encoded_checkers,value):
                        n_checkers = key
                # subtract a checker
                observation[player]['board'][old_pos] = copy(self.encoding[n_checkers-1])

            # 'PLACING DOWN' A CHECKER

            # are we bearing off?
            if new_pos == 24:
                observation[player]['menoff'] += 1/15

            else:
                # get the current number of checkers at the position to which we need to add a checker
                encoded_checkers = observation[player]['board'][new_pos]
                # decode
                for key, value in self.encoding.items():
                    if np.array_equal(encoded_checkers,value):
                        n_checkers = key
                # add a checker
                observation[player]['board'][new_pos] = copy(self.encoding[n_checkers+1])

                # check for blots
                mirror_pos = new_pos+23-2*new_pos
                if not np.array_equal(observation[opponent]['board'][mirror_pos],[0,0,0,0]):
                    # if there is a blot, move the opponent's piece to the bar
                    observation[opponent]['board'][mirror_pos] = [0,0,0,0]
                    observation[opponent]['barmen'] += 0.5

        # swap whose turn it is
        observation['W']['turn'] = 1-observation['W']['turn']
        observation['B']['turn'] = 1-observation['B']['turn']
        
        return self._get_obs(observation)

        
 # W:    0 1 ... 23
# B:      23 ... 1 0

# translating from one board to the other: n+23-(2n)

# idea: replace W and B with 1 and 0