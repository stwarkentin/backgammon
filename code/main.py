#!/usr/bin/env python 3
from network import Network
from agent import TDAgent
from env.backgammon import BackgammonEnv
import numpy as np
import os
from tqdm import tqdm


env = BackgammonEnv()

hidden_units = 0
network = Network(hidden_units)
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
    if (i+1) % 100 == 0:
        np.save("n_moves.npy", moves_per_game)
        filename = "n_units"+str(hidden_units)+"episode"+str(i+1)
        network.save_weights('checkpoints/TDAgent/'+filename)

print("Done")

