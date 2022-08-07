from dice import roll

class Agent:
    def __init__(self, gamma, network):

        self.gamma = gamma
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.network = network
        
    def find_legal_actions(self, obs, dice):
        for die in dice:
            
        return actions

    def choose_action(self, obs):
        obs['W']['board'] = obs['W']['board'].flatten().tolist()
        obs['B']['board'] = obs['B']['board'].flatten().tolist()
        
        dice = roll()
        
        actions = find_legal_actions(obs, dice)
        # insert function that determines possible actions and appends them
        # to actions array
        
        values = [length_of_actions]
        
        # the states are all subsequent states that can be reached based on 
        # the possible actions
        
        for action in actions:
            values[i] = self.network.call(obs+action)
            
        index = np.where(values == values.max())
            
        return actions[index]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        rewards = np.array(self.reward_memory)

        # apply gamma discount factor to rewards
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum


        # get log_prob for all actions in trajectory and calculate loss
        # https://keras.io/api/optimizers/
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state, action) in enumerate(zip(G, self.state_memory, self.action_memory)):
              state = tf.convert_to_tensor([state], dtype=tf.float32)
              actions = [action]
              states = [state]
              log_prob = self.policy.probs_from_states_and_actions(states, actions)
              
              loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)

        # apply gradient doesn't work
        # how to get hold of optimizer?
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
