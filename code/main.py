from network import Network
from agent import TDAgent
from env.backgammon import BackgammonEnv
import numpy as np
from tqdm import tqdm


env = BackgammonEnv()
network = Network()
network.build([1,198])
alpha = 0.1
lmbd = 0.7
gamma = 1
agent = TDAgent(env, network, alpha, lmbd, gamma)
episodes = 50000
moves_per_game = np.asarray([])
for i in tqdm (range(episodes), desc = "Learning..."):
    n_moves = agent.learn()
    np.append(moves_per_game,n_moves)

np.save("n_moves.npy", moves_per_game)
network.save_weights('0checkpoint50000')
print("Done")

