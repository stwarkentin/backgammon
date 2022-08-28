from network import Network
from agent import DQNAgent
from env.backgammon import BackgammonEnv
from memory import ReplayBuffer
import numpy as np

env = BackgammonEnv()
network = Network()
network.build([1,198])
memory = ReplayBuffer(100000)
gamma = 0.99
lr = 0.01
epsilon = 0.9
min_epsilon = 0.1
epsilon_decay = 0.99999
batch_size = 60 
agent = DQNAgent(env, network, gamma, lr, epsilon, min_epsilon, epsilon_decay, memory, batch_size)
episodes = 5
turn_tracker = []
for i in range(episodes):
    obs = env.reset()
    done = False
    turns = 0

    # play the game
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done = env.step(action)
        agent.store_transition(obs, action, reward, done, obs_)
        obs = obs_
        # keep track of turns
        turns += 1
        agent.learn()

    turn_tracker.append(turns)
    print('episode: ', i, 'turns %5d' % turns,
            'reward %.2f' % reward,
            'epsilon %.2f' % agent.epsilon)
    
print("Done")