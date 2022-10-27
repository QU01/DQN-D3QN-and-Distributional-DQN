import numpy as np

class ReplayBuffer:
    
    def __init__(self, max_size, batch_size):
        
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size
    
    def append(self, state, next_state, action, reward, done, info):
        
        sample = (state, next_state, action, reward, done, info)
        
        if len(self.buffer) >= self.max_size:
            
            self.buffer.pop(0)
        
        self.buffer.append(sample)
        
    def sample(self):
        
        idxs = np.random.choice(np.arange(len(self.buffer)), size = self.batch_size)
        
        states = [self.buffer[i][0] for i in idxs]
        next_states = [self.buffer[i][1] for i in idxs]
        actions = [self.buffer[i][2] for i in idxs]
        rewards = [self.buffer[i][3] for i in idxs]
        dones = [self.buffer[i][4] for i in idxs]
            
        return np.array(states), np.array(next_states), actions, rewards, dones