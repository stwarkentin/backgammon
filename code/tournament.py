from network import Network
from agent import TDAgent, DQNAgent, RandomAgent, Agent
from env.backgammon import BackgammonEnv

# 0 , int(5e4) 
# 10 , int(5e4)
# 20 , int(1e5)
# 40 . int(4e5)
hidden_units = [0, 10, 20, 40]
max_episodes = [int(5e4), int(5e4), int(1e5), int(4e5)]

env = BackgammonEnv()
alpha = None
lmbd = None
gamma = None
lr = None
epsilon = 0
min_epsilon = None
epsilon_decay = None
batch_size = None 

# TDAgent:

td_agents = [] 

for i in range(len(hidden_units)):
    network = Network(hidden_units[i])
    network.load_weights('checkpoints/TDAgent'+str(hidden_units[i])+'/episode'+str(max_episodes[i])+'of'+str(max_episodes[i])+'.hdf5')
    td_agents.append(TDAgent(env, network, alpha, lmbd, gamma))

# DQNAgent

dqn_agents = []

for i in range(len(hidden_units)):
    network = Network(hidden_units[i])
    network.load_weights('checkpoints/DQNAgent'+str(hidden_units[i])+'/episode'+str(max_episodes[i])+'of'+str(max_episodes[i])+'.hdf5')
    dqn_agents.append(DQNAgent(env, network, gamma, lr, epsilon, min_epsilon, epsilon_decay, memory, batch_size))

# Random Agent
random_agent = RandomAgent(Agent(env))


# Game blueprint

episodes = 100
white_score = 0
black_score = 0
network = Network(0)
white_agent = random_agent
black_agent = TDAgent(env, network, alpha, lmbd, gamma)

for i in range(episodes):
    # reset board
    obs = env.reset()
    done = False

    # play the game
    while not done:
        if obs['W']['turn'] == 1:
            action = white_agent.choose_action(obs)
        else:
            action = black_agent.choose_action(obs)

        obs_, reward, done = env.step(action)
        
        # 1) White wins, Black is gammoned
        if reward == 2:
            white_score += reward
        # 2) White wins
        elif reward == 1:
            white_score += reward
        # 3) Black wins, White is gammoned
        elif reward == -2: 
            black_score += -reward
        # 4) Black wins
        elif reward == -1:
            black_score += -reward


        obs = obs_

    print("White: ", white_score, " Black: ", black_score)
        
print("Done")