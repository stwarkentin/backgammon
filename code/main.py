from network import Network
from agent import TDAgent
from env.backgammon import BackgammonEnv
from tqdm import tqdm


env = BackgammonEnv()
network = Network()
network.build([1,198])
alpha = 0.1
lmbd = 0.7
gamma = 1
agent = TDAgent(env, network, alpha, lmbd, gamma)
episodes = 50000
for i in tqdm (range(episodes), desc = "Learning..."):
    agent.learn()
network.save_weights('0checkpoint50000')
print("Done")

