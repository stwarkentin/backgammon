import os
import pickle
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
        self.max_length = capacity
        if os.path.exists('memory_buffer.pkl'): # make sure to delete this file between agents!!!!!!!
            self.buffer = pickle.load(open('memory_buffer.pkl', 'rb'))
        else:
            self.buffer = []
            
        print(len(self.buffer))

    def append(self, experience):
        """
        Appends new datapoint to buffer
        
            Parameters:
                    experience (array): datapoint
        
        """
        if len(self.buffer) < self.max_length:
            self.buffer.append(experience)
        else:
            del(self.buffer[0])
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
        
        states = []
        actions = []
        rewards = []
        dones = []
        states_ = []
        
        for i in batch:
            states.append(self.buffer[i][0])
            actions.append(self.buffer[i][1])
            rewards.append(self.buffer[i][2])
            dones.append(self.buffer[i][3])
            states_.append(self.buffer[i][4])
        
        return states, actions, rewards, dones, states_
    
    def save(self):
        pickle.dump(self.buffer, open('memory_buffer.pkl', 'wb'))