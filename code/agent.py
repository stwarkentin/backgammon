from dice import roll
from random import choice
from copy import deepcopy
from human_readable import human_readable

class Agent:
    def __init__(self, env, network):
        self.env = env
        self.network = network        

    def find_actions(self, obs):
        
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
        player_obs['board'] = human_readable(player_obs)
        
        # !!! just for testing purposes !!!
        # if not length == size and not len(action) == 0:
        #test_opponent = deepcopy(opponent_obs)
        #new_op_board = []
        #for idx, pos in enumerate(test_opponent['board']):
        #    new_op_board.append(pos[0] + pos[1] + pos[2] + pos[3] * 2)
        #test_opponent['board'] = new_op_board
        #print("Dice: ", dice)
        #print("Player:\t\t", player, player_obs['board'])
        #print("Opponent:\t", opponent, test_opponent['board'])
        
        # recursive function to search "action-tree"
        def build_actions(action, dice, player_obs):
            
            # if we have iterated through all dice the action is appended and we return
            if len(dice) == 0:
                legal_actions.append(action)
                return

            # in case there are chips in the bar they have to be removed before any other actions can be taken
            if player_obs['barmen'] > 0:
                # check for free points/blots
                for i, die in enumerate(dice):
                    # for each die, check if the barmen would land on a free spot/a blot
                    if opponent_obs['board'][die + 23 - 2 * die][1] == 0:
                        # create a new observation in which the barman has been freed
                        new_player_obs = deepcopy(player_obs)
                        new_player_obs['board'][die] += 1
                        new_player_obs['barmen'] -= 0.5
                        # remove the used die
                        new_dice = copy(dice)
                        new_dice.pop(i)
                        # append the chosen move to action, pass new observation and dice to the recursion
                        build_actions(action.copy() + [(-1, die)], new_dice, new_player_obs)
                        
            else:
                # is it legal to move off the board?
                bearingoff = True
                # check for checkers in the first three quadrants
                # NOTE: easier to just add up all the values and compare them to zero?
                for idx in range(18):
                    if player_obs['board'][idx] > 0:
                        bearingoff = False
                        break
                        
                # iterate through all positions and check if we can move to position + current dice
                for pos, n_checkers in enumerate(player_obs['board']):
                    # check if a dice roll could exceed the limitations of the board
                    if n_checkers > 0 and (pos + dice[0]) > 23:
                        # can a checker be moved off the board
                        if bearingoff:
                            if pos+dice[0] == 24 or sum(player_obs['board'][:pos]) == 0:
                                # create a new observation
                                new_player_obs = deepcopy(player_obs)
                                new_player_obs['board'][pos] -= 1
                                # recursively call function
                                build_actions(action.copy() + [(pos, 24)], dice[1:], new_player_obs)       
                            
                    # If there is no checker on the point indicated by the roll, the player must make a legal move using a checker on a higher-numbered point. If there are no checkers on higher-numbered points, the player is permitted (and required) to remove a checker from the highest point on which one of his checkers resides.
                    
                    # move a checker only if the move is legal
                    elif n_checkers > 0 and opponent_obs['board'][(pos + dice[0]) + 23 - 2 * (pos + dice[0])][1] == 0:
                        new_player_obs = deepcopy(player_obs)
                        new_player_obs['board'][pos] -= 1
                        new_player_obs['board'][pos + dice[0]] += 1
                        # recursively call function
                        build_actions(action.copy() + [(pos, pos + dice[0])], dice[1:], new_player_obs)
                            
            # if we couldn't move and reach return we recursively call the function with the same state and action but iterated dice 
            build_actions(action.copy(), dice[1:], player_obs)
    
        # call the recursive function
        build_actions(action, dice, player_obs)
        
        # only keep actions if they have the max length
        length = max(len(x) for x in legal_actions)
        legal_actions = list(l for l in legal_actions if len(l) == length)

        return legal_actions

class RandomAgent(Agent):

    # # If the __init__ method is not defined in a child class then the __init__ method from the parent class is automatically used.
    # def __init__(self,env,network):
    #     super.__init__(env,network)

    def choose_action(self,obs):

        legal_actions = self.find_actions(obs)

        afterstates = [] 
        # call function that returns the state
        for action in legal_actions:
            state = env.step(False, action)
            afterstates.append(state.copy())

        values = []
        for state in afterstates:
            values.append(self.network.call(state.reshape(1,-1)))
            
        print(values[0][0][0])
           
        action = choice(legal_actions)
        
        print("Action: ", action)
            
        return action

class TDAgent(Agent):

    def __init__(self, env, network, alpha, lmbd, gamma):
        super.__init__(env,network)
        self.alpha = alpha
        self.lmbd = lmdd
        self.gamma = gamma

    def choose_action(self, obs):
        pass
    
    def learn(self):
        pass

        # !!! missing something that evaluates at which value index we have the best probability of winning !!!
        # example output: [.1,.2,.4,.3] 0.1+2*.02-0.4-2*0.3 = -0.5
        # maybe using numpy arrays??
        # !!! PLACEHOLDER !!!

        # values = []
        # for state in states:
        # value = self.network.call(state.reshape(1,-1))[0]
        # if player == 'W':
        # values.append(float(value[0] + 2 * value[1] - value[2] - 2 * value[3]))
        # else:
        # values.append(float(-value[0] - 2 * value[1] + value[2] + 2 * value[3]))

        # action = legal_actions[values.index(max(values))]   
    
        

