import numpy as np
import matplotlib.pyplot as plt

from network import Network
from agent import TDAgent, DQNAgent, RandomAgent, Agent
from env.backgammon import BackgammonEnv

env = BackgammonEnv()
alpha = None
lmbd = None
gamma = None
lr = None
epsilon = 0
min_epsilon = None
epsilon_decay = None
batch_size = None 
memory = None

# TDAgent: 
td_50000_network = Network()
td_50000_network.build([1,198])
td_50000_network.load_weights('checkpoints/TDAgent0episode50000.hdf5')
td_50000_agent = TDAgent(env, td_50000_network, alpha, lmbd, gamma)

td_2000_network = Network()
td_2000_network.build([1,198])
td_2000_network.load_weights('checkpoints/TDAgent0episode2000.hdf5')
td_2000_agent = TDAgent(env, td_2000_network, alpha, lmbd, gamma)

# DQNAgent
dqn_network = Network()
dqn_network.build([1,198])
dqn_network.load_weights('checkpoints/DQNAgent0episode2000.hdf5')
dqn_agent = DQNAgent(env, dqn_network, gamma, lr, epsilon, min_epsilon, epsilon_decay, memory, batch_size)

# Random Agent
random_agent = RandomAgent(Agent(env))


# Game blueprint
episodes = 1000
score = [0]
white_score = 0
black_score = 0
x = [0, 0, 0, 0]

for i in range(episodes):
    # reset board
    obs = env.reset()
    done = False

    # play the game
    while not done:
        if obs['W']['turn'] == 1:
            action = td_2000_agent.choose_action(obs)
        else:
            action = random_agent.choose_action(obs)

        obs_, reward, done = env.step(action)
        
        # 1) White wins, Black is gammoned
        if reward == 2:
            white_score += reward
            x[0] += 1
        # 2) White wins
        elif reward == 1:
            white_score += reward
            x[1] += 1
        # 3) Black wins, White is gammoned
        elif reward == -2: 
            black_score += -reward
            x[2] += 1
        # 4) Black wins
        elif reward == -1:
            black_score += -reward
            x[3] += 1
            
        score.append(score[-1] + reward)
        obs = obs_

        
plt.style.use('_mpl-gallery-nogrid')

def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# plot
labels = ['TD gammons', 'TD wins', 
'Random gammons', 'Random wins']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(x, colors=colors, radius=3, autopct=lambda pct: func(pct, x), center=(4, 4),
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True, textprops=dict(color="w"))

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))
ax.set_title('TD-agent vs. Random agent (1000 episodes)')
ax.legend(patches, labels, loc="best")

plt.show()
