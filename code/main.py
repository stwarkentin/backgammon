from network import Network
from agent import TDAgent
from env.backgammon import BackgammonEnv


env = BackgammonEnv()
network = Network()
network.build([1,198])
alpha = 0.1
lmbd = 0.7
gamma = 1
agent = TDAgent(env, network, alpha, lmbd, gamma)
episodes = 3
for i in range(episodes):
    agent.learn()
print("Done")

