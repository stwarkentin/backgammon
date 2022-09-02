from network import Network
from agent import DQNAgent
from env.backgammon import BackgammonEnv
from memory import ReplayBuffer
import numpy as np
from tqdm import tqdm

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

episodes = 50000
moves_per_game = np.asarray([])


for i in tqdm (range(episodes), desc = "Learning..."):
    n_moves = agent.learn()
    np.append(moves_per_game,n_moves)
    if (i+1) % 100 == 0:
        np.save("n_moves.npy", moves_per_game)
        filename = "n_units"+str(hidden_units)+"episode"+str(i+1)
        network.save_weights('checkpoints/TDAgent/'+filename)

print("Done")

    