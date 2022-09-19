import numpy as np
from tqdm import tqdm

from network import Network
from agent import DQNAgent
from env.backgammon import BackgammonEnv
from memory import ReplayBuffer

def main():
    env = BackgammonEnv()

    # build the model
    network = Network()
    network.build([1,198])
    
    completed_episodes = 0
    episodes = 60

    # inititate the agentalpha = 0.1
    memory = ReplayBuffer(100000)
    gamma = 0.99
    lr = 0.01
    epsilon = 0.9
    min_epsilon = 0.1
    epsilon_decay = 0.99999
    batch_size = 60 
    agent = DQNAgent(env, network, gamma, lr, epsilon, min_epsilon, epsilon_decay, memory, batch_size)


    for i in tqdm (range(episodes), desc = "Learning..."):
        agent.learn()
        completed_episodes += 1
        print(completed_episodes)
        
if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', "output.dat")
    
    import pstats
    from pstats import SortKey
    
    with open("output_time_dqn_fix.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
        
    with open("output_calls_dqn_fix.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()