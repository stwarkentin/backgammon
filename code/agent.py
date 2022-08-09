from dice import roll
from random import choice
from copy import deepcopy

class Agent:
    def __init__(self, env, gamma, lam, network):
        self.env = env
        self.lam = lam
        self.gamma = gamma
        self.network = network        

    def choose_action(self, obs):
        
        dice = roll()
        legal_actions = []
        size = len(dice)
        action = []

        # check whose turn it is
        if obs['W']['turn'] == 1:
            player = 'W'
            opponent = 'B'
        else:
            player = 'B'
            opponent = 'W'
            
        player_obs = deepcopy(obs[player])
        opponent_obs = obs[opponent]
        
        # make board more easily readable
        new_board = []
        for pos in player_obs['board']:
            new_board.append(pos[0] + pos[1] + pos[2] + pos[3] * 2)
        player_obs['board'] = new_board
        
        # recursive function to search "action-tree"
        def find_board_actions(action, dice, player_obs):
            
            # if we have iterated through all dice the action is appended and we return
            if len(dice) == 0:
                legal_actions.append(action)
                return

            # in case there are chips in the bar they have to be removed before any other actions can be taken
            if player_obs['barmen'] > 0:
                for i, die in enumerate(dice):
                    if opponent_obs['board'][die + 23 - 2 * die][1] == 0:
                        new_player_obs = deepcopy(player_obs)
                        new_player_obs['board'][die] += 1
                        new_player_obs['barmen'] -= 0.5
                        find_board_actions(action.copy() + [(-1, die)], dice[1:], new_player_obs)
            else:
                # is it legal to move off the board?
                moveoff = True
                for idx in range(18):
                    if player_obs['board'][idx] > 1:
                        moveoff = False
                        break
                        
                # iterate through all poitions and check if we can move to position + current dice
                for idx, pos in enumerate(player_obs['board']):
                    if pos > 0 and (idx + dice[0]) > 23:
                        if moveoff:
                            new_player_obs = deepcopy(player_obs)
                            new_player_obs['board'][idx] -= 1
                            # recursively call function
                            if len(dice) >= 1:
                                find_board_actions(action.copy() + [(idx, 24)], dice[1:], new_player_obs)

                    elif pos > 0 and opponent_obs['board'][(idx + dice[0]) + 23 - 2 * (idx + dice[0])][1] == 0:
                        new_player_obs = deepcopy(player_obs)
                        new_player_obs['board'][idx] -= 1
                        new_player_obs['board'][idx + dice[0]] += 1
                        # recursively call function
                        if len(dice) >= 1:
                            find_board_actions(action.copy() + [(idx, idx + dice[0])], dice[1:], new_player_obs)
                            
            # if we couldn't move and reach return we recursively call the function with the same state and action but iterated dice 
            find_board_actions(action.copy(), dice[1:], player_obs)
    

        find_board_actions(action, dice, player_obs)
        
        # only keep actions if they have the max length
        length = max(len(x) for x in legal_actions)
        legal_actions = list(l for l in legal_actions if len(l) == length)
          
        states = [] 
        # call function that returns the state
        for action in legal_actions:
            state = env.get_state(deepcopy(obs), action)
            states.append(state.copy())

        values = []
        for state in states:
            values.append(self.network.call(state))

        # !!! missing something that evaluates at which value index we have the best probability of winning !!!
        # maybe using numpy arrays??
        # !!! PLACEHOLDER !!!
        action = choice(legal_actions)
        return action

    def learn(self):
        pass