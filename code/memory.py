import numpy as np
from collections import deque 

class ReplayBuffer:
    
    def __init__(self, capacity):
          self.buffer = deque(maxlen = capacity)

    def append(self, experience):
          self.buffer.append(experience)

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = [self.buffer[i][0] for i in batch]
        actions = [self.buffer[i][1] for i in batch]
        rewards = [self.buffer[i][2] for i in batch]
        dones = [self.buffer[i][3] for i in batch]
        states_ = [self.buffer[i][4] for i in batch]
        
        return states, actions, rewards, dones, states_