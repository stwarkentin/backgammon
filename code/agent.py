from dice import roll
from random import choice
from copy import copy, deepcopy
from human_readable import human_readable
import tensorflow as tf

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

    def whose_turn(self,obs):
        
        if obs['W']['turn'] == 1:
            player = 'W'
            opponent = 'B'
        else:
            player = 'B'
            opponent = 'W'

        return player, opponent

    def score(self,state):

        player, opponent = self.whose_turn(state)

        state = self.env._flatten_obs(state)
        value = self.network.call(state.reshape(1,-1))[0]

        if player == 'W':
            score = float(value[0] + 2 * value[1] - value[2] - 2 * value[3])
        else:
            score = float(-value[0] - 2 * value[1] + value[2] + 2 * value[3])
        
        return score

class RandomAgent(Agent):

    # # If the __init__ method is not defined in a child class then the __init__ method from the parent class is automatically used.
    # def __init__(self,env,network):
    #     super.__init__(env,network)

    def choose_action(self,obs):

        legal_actions = self.find_actions(obs)
            
        return choice(legal_actions)

class TDAgent(Agent):

    def __init__(self, env, network, alpha, lmbd, gamma):
        super().__init__(env,network)
        self.alpha = alpha
        self.lmbd = lmbd
        self.gamma = gamma
        # self.state = []
        # self.post_state = []
        # self.reward = 0
        # self.z = np.zeros(self.network.weight_shape)

    def choose_action(self, obs):

        player, opponent = self.whose_turn(obs)
        legal_actions = self.find_actions(obs)

        afterstates = [] 
        # call function that returns the state
        for action in legal_actions:
            state = self.env.step(False, action)
            afterstates.append(state.copy())
        scores = []
        for state in afterstates:
            scores.append(self.score(state))

        action = legal_actions[scores.index(max(scores))] 
        print("Action: ", action)
            
        return action

    # def store_transition(self, state, post_state, reward):
    #     self.state = state
    #     self.post_state = post_state
    #     self.reward = reward
    
    # play one episode, learn
    def learn(self):
        # reset the board
        # do this in main?
        obs = self.env.reset()
        done = False
        # initialize eligibility trace
        z = []
        w = self.network.trainable_weights
        for layer in w:
            z.append(tf.Variable(tf.zeros_like(layer)))

        print("Empty z:",z)
        
        # play the game
        while not done:
            # choose an action, observe outcome
            action = self.choose_action(obs)
            obs_, reward, done = self.env.step(True, action)

            # get the value gradient so that we can update the eligibility trace
            with tf.GradientTape() as tape: # 'with...as...' automatically closes the GradientTape object at the end of the block
                state = self.env._flatten_obs(obs)
                value = self.network(state.reshape(1,-1))
            gradients = tape.gradient(value, w)

            print("One Gradient:",gradients)

                value = self.network(obs)
            gradients = tape.gradient(value, w)


            # update eligibility trace
            for z_, gradient in zip(z, gradients):
                z_.assign(self.gamma * self.lmbd * z_ + gradient)

            print("z:",z)

            # TD error
            if done:
                target = reward
            else:
                state_ = self.env._flatten_obs(obs_)
                target = reward + self.gamma *  self.network(state_.reshape(1,-1))
            delta = target - self.network(state.reshape(1,-1))

                target = reward + self.gamma *  self.network(obs_)
            delta = target - self.network(obs)


            print("TD error:", delta)

            # update weights

            for w_, z_ in zip(w, z):
                w.assign_add(tf.reshape(self.alpha * delta * z_, w_.shape)) # 'w.assign_add' = 'w+...'

            obs = obs_

        # # get and flatten weights
        # w = []
        #     for layer in self.list_of_layers:
        #         w.append(layer.get_weights())
        #     w = weights.flatten()

        #     if len(self.network.list_of_layers) == 1:
        #         two_layers = True 
        #     else:
        #         two_layers = False

        #     # update the weights vector
        #     w = w + self.alpha*delta*self.z

        #     # set the weights (split by layer)
        #     if two_layers:
        #         self.network.hidden_layer.set_weights(:-4)
        #         self.network.output_layer.set_weights(-4:)
        #     else:
        #         self.network.output_layer.set_weights(w)

class DQNAgent(Agent):
    pass
        