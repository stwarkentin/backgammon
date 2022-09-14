from dice import roll
from human_readable import human_readable

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np

from random import choice
from copy import copy, deepcopy

class Agent:
    """Parent class from which all other agent classes inherit.

    ...

    Attributes
    ----------
    env : object
        custom backgammon gym environment
    network : object
        artificial neural network which learns the policy

    Methods
    -------
    find_actions(obs):
        Returns all legal actions given an observation

    whose_turn(obs):
        Returns current player and opponent given an observation

    score(state):
        Evaluates the current state/observation? from the perspective of the current player
    """
    def __init__(self, env, network = None):
        self.env = env
        self.network = network        

    def find_actions(self, obs):
        """
        Returns all legal actions given an observation

            Parameters:
                    obs (dict): current state

            Returns:
                    legal_actions (array): contains all legal actions for the given state
        """
        # fetch game-relevant information
        player, opponent = self.whose_turn(obs)
        dice = roll()
        
        # split the observation into player and opponent boards
        player_obs = deepcopy(obs[player])
        opponent_obs = obs[opponent]
        
        # decode the board to simplify addition and subtraction
        player_obs['board'] = human_readable(player_obs)

        legal_actions = []
        
        # recursive function to search "action-tree"
        def build_actions(action, dice, player_obs):
            
            # check if we have iterated through all dice. if so, the action is appended to the list of legal actions and we return
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
    
        # 
        action = []
        
        # call the recursive function
        build_actions(action, dice, player_obs)
        
        # only keep actions if they have the max length
        length = max(len(x) for x in legal_actions)
        legal_actions = list(l for l in legal_actions if len(l) == length)

        return legal_actions

    def whose_turn(self, obs):
        """
        Returns current player and opponent given an observation
            
            Parameters:
                    obs (dict): current state
            
            Returns:
                    player (str): key for current player
                    opponent (str): key for current opponent
        
        """
        if obs['W']['turn'] == 1:
            player = 'W'
            opponent = 'B'
        else:
            player = 'B'
            opponent = 'W'

        return player, opponent

    def score(self, player, state):
        """
        Evaluates the current state, using the network, from the perspective of the current player
        
            Parameters:
                    player (str): key for current player
                    state (dict): current state
            
            Returns:
                    score (float): player dependently weighted sum of winning probabilities
        """
        value = self.network.call(state.reshape(1,-1))[0]

        # create player dependent weighted sum of probabilities of 4 different outcomes
        if player == 'W':
            score = float(value[0] + 2 * value[1] - value[2] - 2 * value[3])
        else:
            score = float(-value[0] - 2 * value[1] + value[2] + 2 * value[3])
        
        return score

class RandomAgent(Agent):
    """Random agent class.

    ...

    Methods
    -------
    choose_actions(obs):
        Chooses a random action

    """
    def choose_action(self,obs):
        """
        Chooses random action

            Parameters:
                    obs (dict): current state

            Returns:
                    action (array): random action
        """
        legal_actions = self.find_actions(obs)
        action = choice(legal_actions)
        return action

class TDAgent(Agent):
    """TDAgent that implements td-lambda algorithm 

    ...
    
    Attributes
    ----------
    alpha : float
        step-size
    _lambda : float
        lambda weighting factor
    gamma : float
        discount factor

    Methods
    -------
    choose_action(obs):
        Chooses action that results in the state with the highest score for the current player

    learn():
        Plays the game and trains the network on sampled batch
    """

    def __init__(self, env, network, alpha, _lambda, gamma):
        super().__init__(env,network)
        self.alpha = alpha
        self._lambda = _lambda
        self.gamma = gamma

    def choose_action(self, obs):
        """
        Chooses action that results in the state with the highest score for the current player

            Parameters:
                    obs (dict): current observation

            Returns:
                    action (array): legal action that maximizes the score for the next state
        """
        player, opponent = self.whose_turn(obs)
        legal_actions = self.find_actions(obs)

        scores = [] 
        # iterate through all actions and collect the post-state score
        for action in legal_actions:
            obs_ = self.env.simulate_step(obs, action)
            state = self.env._flatten_obs(obs_)
            scores.append(self.score(player, state))

        # assign action that maximizes score
        action = legal_actions[scores.index(max(scores))] 
            
        return action

    def learn(self):
        """
        Plays the game and trains the network on sampled batch
            
            Returns:
                    n_moves (int): number of moves that were made during the game
        """
        # reset the board and movecounter
        obs = self.env.reset()
        done = False
        n_moves = 0
        
        # initialize eligibility trace
        w = self.network.trainable_weights
        z = []
        for layer in w:
            z.append(tf.Variable(tf.zeros_like(layer)))

        # play the game
        while not done:
            # choose an action, observe outcome
            action = self.choose_action(obs)
            obs_, reward, done = self.env.step(action)

            # get the value gradient so that we can update the eligibility trace
            with tf.GradientTape() as tape:
                state = self.env._flatten_obs(obs)
                value = self.network(state.reshape(1,-1))
            gradients = tape.gradient(value, w)

            # update eligibility trace
            for z_, gradient in zip(z, gradients):
                z_.assign(self.gamma * self._lambda * z_ + gradient)

            # TD error
            if done:
                target = reward
            else:
                state_ = self.env._flatten_obs(obs_)
                target = reward + self.gamma *  self.network(state_.reshape(1,-1))
            delta = target - self.network(state.reshape(1,-1))

            # update weights
            for w_layer, z_layer in zip(w, z):
                w_layer.assign_add(tf.reshape(self.alpha * delta * z_layer,w_layer.shape))
            
            # update observation and movecounter
            obs = obs_
            n_moves += 1


        return n_moves
            
class DQNAgent(Agent):
    """Agent class that implements the dqn algorithm
    
    ...

    Attributes
    ----------
    gamma : float
        discount factor
    epsilon : float
        probability for taking a random action
    min_epsilon : float
        lowest value of epsilon
    epsilon_decay : float
        epsilon discount factor
    memory : object
        instance of a replay buffer to store and sample data
    batch_size : int
        length of sample data the network uses to learn

    Methods
    -------
    store_transition(state, action, reward, done, new_state):
        Appends state transition to memory

    choose_action(obs, random):
        Chooses random action or action that results in the state with the highest value for the current player based on epsilon

    learn():
        Plays the game, trains the network on sampled batch and decays epsilon
    """
    def __init__(self, env, network, gamma, lr, epsilon, min_epsilon, epsilon_decay, memory, batch_size):
        super().__init__(env, network)
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = memory
        self.batch_size = batch_size
        self.network.compile(optimizer = Adam(learning_rate = lr), loss = 'mean_squared_error')

    def store_transition(self, state, action, reward, done, new_state):
        """
        Appends state transition to memory
        
             Parameters:
                    state (dict): previous state
                    action (array): action that was taken
                    reward (int): reward that was received
                    new_state (dict) : new state that was reached

        """
        self.memory.append([state, action, reward, done, new_state])

      
    def choose_action(self, obs, random = True):
        """
        Chooses random action or action that results in the state with the highest value for the current player based on epsilon
        
            Parameters:
                    obs (dict): current observation
                    random (bool): mode with epsilon or without
            
            Returns:
                    action (array): legal action that either is chosen at random or maximizes the score for the next state
        """
        player, opponent = self.whose_turn(obs)
        legal_actions = self.find_actions(obs)
        
        # choose random action or "next-state-value" maximizing action with odds epsilon : 1-epsilon
        if np.random.random() < self.epsilon and random:
            action = choice(legal_actions)
            
        else:
            scores = [] 
            # iterate through all actions and collect the post-state score
            for action in legal_actions:
                obs_ = self.env.simulate_step(obs, action)
                state = self.env._flatten_obs(obs_)
                scores.append(self.score(player, state))

            # assign action that maximizes score
            action = legal_actions[scores.index(max(scores))] 

        return action

    def learn(self):
        """
        Plays the game, trains the network on sampled batch and decays epsilon
            
            Returns:
                    n_moves (int): number of moves that were made during the game
        """
        # reset board and movecounter
        obs = self.env.reset()
        done = False
        # n_moves = 0

        # play the game
        while not done:
            # choose an action, observe outcome, store observation
            action = self.choose_action(obs)
            obs_, reward, done = self.env.step(action)
            self.store_transition(obs, action, reward, done, obs_)

            # if our buffer is not filled sufficiently return, else train
            if len(self.memory.buffer) < self.batch_size:
                return

            # sample batch
            states, actions, rewards, dones, states_ = self.memory.sample(self.batch_size)

            flat_states = []
            target = []

            # build targets
            for i in range(self.batch_size):
                flat_states.append(self.env._flatten_obs(states[i]))

                if dones[i]:
                    target.append(rewards[i] * tf.zeros(4,))
                else:
                    # find the max state-action value of the subsequent state (action that maximzes the sub-subsequent state value)
                    state__ = self.env.simulate_step(states_[i], self.choose_action(states_[i], False))
                    state__ = self.env._flatten_obs(state__)
                    target.append(rewards[i] + self.gamma * self.network(state__.reshape(1,-1))[0])

            flat_states = np.array(flat_states)
            target = np.array(target)

            # train network on target batch
            self.network.train_on_batch(flat_states, target)

            # decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)
            
            # update observation and movecounter
            obs = obs_
            # n_moves += 1

        return #n_moves