import numpy as np
from collections import deque 

class ReplayBuffer:
    """Replay buffer that stores and samples from data
    ...

    Attributes
    ----------
    buffer : deque
        deque that stores all data

    Methods
    -------
    append(experience):
        Appends new datapoint to buffer
    sample(batch_size):
        Returns sample with random datapoints of length batch_size

    """
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def append(self, experience):
        """
        Appends new datapoint to buffer
        
            Parameters:
                    experience (array): datapoint
        
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Returns sample with random datapoints of length batch_size
            
            Parameters:
                    batch_size (int): length of sample
            
            Returns:
                states (array): array with sampled states
                actions (array): array with sampled actions
                rewards (array): array with sampled rewards
                dones (array): array with sampled dones
                states_ (array): array with sampled states_
    
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = [self.buffer[i][0] for i in batch]
        actions = [self.buffer[i][1] for i in batch]
        rewards = [self.buffer[i][2] for i in batch]
        dones = [self.buffer[i][3] for i in batch]
        states_ = [self.buffer[i][4] for i in batch]
        
        return states, actions, rewards, dones, states_