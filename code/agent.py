from dice import roll
class Agent:
    def __init__(self, gamma, network, env):
        self.env = env
        self.gamma = gamma
        self.network = network

    def choose_action(self, obs):
        
        dice = roll()
        
        legal_actions = []
        size = len(dice)
        action = []

        # check whose turn it is
        if obs['W']['turn'] == 1:
            player_obs = obs['W']
            opponent_obs = obs['B']
        else:
            player_obs = obs['B']
            opponent_obs = obs['W']
            
        # make board more easily readable
        new_board = []
        for idx, pos in enumerate(player_obs['board']):
            new_board.append(pos[0] + pos[1] + pos[2] + pos[3] * 2)
        player_obs['board'] = new_board
        
        def find_board_actions(action, dice, player_obs):
            # if our action has reached the desired size it is appended and we return
            # !!! problem: what if we cannot use all the dice?? We still need to find legal actions !!!
            if len(action) == size:
                legal_actions.append(action)
                return

            # in case there are chips in the bar they have to be removed before any other actions can be taken
            if player_obs['barmen'] > 0:
                for i, die in enumerate(dice):
                    if opponent_obs['board'][die + 25 - 2 * die][1] == 0:
                        new_player_obs = player_obs.copy()
                        new_player_obs['board'][die] += 1
                        new_player_obs['barmen'] -= 1
                        find_board_actions(action.copy() + [(0, die)], dice[1:], new_player_obs)

            # is it legal to move off the board?
            moveoff = True
            for idx in range(18):
                if player_obs['board'][idx] == 1:
                    moveoff = False
                    break

            for idx, pos in enumerate(player_obs['board']):
                if pos > 0 and idx + dice[0] > 23:
                    if moveoff:
                        new_player_obs = player_obs.copy()
                        new_player_obs['board'][idx] -= 1
                        find_board_actions(action.copy() + [(idx + 1, 25)], dice[1:], new_player_obs)

                if pos > 0 and opponent_obs['board'][(idx + dice[0]) + 24 - 2 * (idx + dice[0])][1] == 0:
                    new_player_obs = player_obs.copy()
                    new_player_obs['board'][idx] -= 1
                    new_player_obs['board'][idx + dice[0]] -= 1
                    find_board_actions(action.copy() + [(idx + 1, idx + dice[0] + 1)], dice[1:], new_player_obs)

        find_board_actions(action, dice, player_obs)
        
        # test print out looking good for initial turn
        print(legal_actions)
        
        states = [] 
        # call function that returns the state 
        for action in legal_actions:
            states.append(env.state_from_action(obs, action))
        
        values = []
        for state in states:
            state['W']['board'] = state['W']['board'].flatten().tolist()
            state['B']['board'] = state['B']['board'].flatten().tolist()
            values.append(self.network.call(state))
            
        # !!! missing something that evaluates at which value index we have the best probability of winning !!!
        # maybe using numpy arrays??
        
        return legal_actions[index]

    def learn(self):
        pass