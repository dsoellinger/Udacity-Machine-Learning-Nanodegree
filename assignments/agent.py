import numpy as np
from collections import defaultdict

# Using the Sarsa algorithm to train the agent

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        
        Params
        ======
        - nA: number of actions available to the agent
        """
        
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = 1.0
        self.alpha = 0.20
        self.epsilon=0.01
       
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # Epsilon greedy policy
        action_max = np.argmax(self.Q[state])
        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[action_max] = 1 - self.epsilon + self.epsilon / self.nA
        
        action = np.random.choice(a=np.arange(self.nA), p=policy)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        # We take an action A_t and observe reward R_(t+1) + next_state S_(t+1)
        # Choose action A_(t+1) using policy derived from Q
        next_action = self.select_action(next_state)
        
        # Update the action-value estimate
        self.Q[state][action] =  self.Q[state][action] + (self.alpha * (reward + (self.gamma * self.Q[next_state][next_action]) - self.Q[state][action]))
        

