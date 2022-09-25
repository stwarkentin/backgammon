import cProfile
import pstats
from pstats import SortKey

from network import Network
from agent import DQNAgent
from env.backgammon import BackgammonEnv
from memory import ReplayBuffer

def main():
    
    # inititate either DQN-agent
    env = BackgammonEnv()
    network = Network()
    gamma = 0.99
    lr = 0.01
    epsilon = 0.9
    min_epsilon = 0.1
    epsilon_decay = 0.99999
    memory = ReplayBuffer(100000)
    batch_size = 20 
    episodes = 10
    agent = DQNAgent(env, network, gamma, lr, epsilon, min_epsilon, epsilon_decay, memory, batch_size)

    # training loop
    for i in range(episodes):
        agent.learn()

    agent.memory.save()
        
if __name__ == "__main__":
    
    # run main() and colect profiling date
    cProfile.run('main()', "output.dat")
    
    # sort data by amount of time
    with open("DQN_10_output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()
        
    # sort data by amount of calls
    with open("DQN_10_output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()