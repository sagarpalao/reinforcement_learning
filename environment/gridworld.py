import numpy as np
import random

class GridWorld687():

    def __init__(self):
        self.goal_state = [(4, 4)]
        self.obstacle_state = [(2, 2), (3, 2)]
        self.water_state = [(4, 2)]
        self.state = None
        self.count = 0

    def reset(self, seed=None):
        self.state = (0, 0)
        self.count = 0
        return (np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]), None)

    def is_invalid_state(self, s):
        if s in self.obstacle_state or s[0] < 0 or s[0] > 4 or s[1] < 0 or s[1] > 4:
            return True
        return False

    def step(self, a):
        self.count = self.count + 1
        s = self.state
        next_states = [(s[0] - 1, s[1]), (s[0], s[1] + 1), (s[0] + 1, s[1]), (s[0], s[1] - 1), s]
        next_states_probability = [0, 0, 0, 0, 0]
        next_states_probability[a] = 0.8
        act = a - 1
        if a - 1 < 0:
            act = 3
        next_states_probability[act] = 0.05
        act = a + 1
        if a + 1 > 3:
            act = 0
        next_states_probability[act] = 0.05
        next_states_probability[4] = 0.1
        next_state = random.choices(next_states, weights=next_states_probability, k=1)[0]
        if self.is_invalid_state(next_state):
            next_state = s
        self.state = next_state
        terminated = self.is_terminal_state(next_state)
        reward = self.get_reward(next_state)
        next_state_one_hot = np.zeros(10)
        next_state_one_hot[next_state[0]] = 1
        next_state_one_hot[5 + next_state[1]] = 1
        return next_state_one_hot, reward, terminated, self.count > 100, None

    def get_reward(self, s):
        if s in self.goal_state:
            return 10
        elif s in self.water_state:
            return -10
        else:
            return 0

    def is_terminal_state(self, s):
        if s in self.goal_state:
            return True
        else:
            return False
