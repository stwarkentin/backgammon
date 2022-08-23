from network import Network
from agent import TDAgent
from env.backgammon import BackgammonEnv


env = BackgammonEnv()
network = Network()
alpha = 0.1
lmbd = 0.7
gamma = 1
agent = TDAgent(env, network, alpha, lmbd, gamma)
episodes = 3
for i in range(episodes):
    agent.learn()
print("Done")
    # done = False
    # observation = env.reset()
    # while not done:
    #     action = agent.choose_action(observation)
    #     obs_, reward, done = env.step(True, action)
    #     agent.store_transition(ons, obs_, reward)
    #     agent.learn()