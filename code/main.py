from network import Network
from agent import TDAgent
from env.backgammon import BackgammonEnv


env = BackgammonEnv()
network = Network()
alpha = 0.001
lmbd = 0.7
gamma = 1
agent = TDAgent(env, network, alpha, lmbd, gamma)
episodes = 50
for i in range(episodes):
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        obs_, reward, done = env.step(True, action)
        agent.store_transition(ons, obs_, reward)
        agent.learn()