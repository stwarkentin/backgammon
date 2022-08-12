import env.backgammon as backgammon
from agent import Agent
from network import network

env = BackgammonEnv()
network = Network()
agent = Agent(env, network)

done = False
observation = env.reset()
while not done:
    action = agent.choose_action(observation)
    observation, reward, done = env.step(True, action)
    
print(reward)



