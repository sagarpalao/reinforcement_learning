import torch
import torch.nn as nn
from torch import optim
import numpy as np
from utils.adapt_lr import AdaptLearningRate

# Multilayer Neural Network with ReLU activation to represent Policy
class PolicyNeuralNetwork(nn.Module):

    def __init__(self, no_of_states, hidden_units, no_of_actions):
        super(PolicyNeuralNetwork, self).__init__()
        layers = []
        for i, k in enumerate(hidden_units):
            layers.append(nn.Linear(no_of_states if i == 0 else hidden_units[i - 1], k))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], no_of_actions))
        layers.append(nn.Softmax(dim=1))
        self.sequence_linear_relu = nn.Sequential(*layers)

    def forward(self, x):
        out = self.sequence_linear_relu(x)
        return out


# Multilayer Neural Network with ReLU activation to represent State-Value Function
class ValueFunctionNeuralNetwork(nn.Module):

    def __init__(self, no_of_states, hidden_units):
        super(ValueFunctionNeuralNetwork, self).__init__()
        layers = []
        for i, k in enumerate(hidden_units):
            layers.append(nn.Linear(no_of_states if i == 0 else hidden_units[i - 1], k))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.sequence_linear_relu = nn.Sequential(*layers)  

    def forward(self, x):
        out = self.sequence_linear_relu(x)
        return out


# Reinforce with Baseline Method
class ReinforceWithBaseline():

    def __init__(self, no_of_states, hidden_units, no_of_actions, alpha_policy, alpha_value):
        # Create neural network to represent policy and state-value function
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = PolicyNeuralNetwork(no_of_states, hidden_units, no_of_actions).to(self.device)
        self.v_hat = ValueFunctionNeuralNetwork(no_of_states, hidden_units).to(self.device)
        self.policy_lr_adapter = AdaptLearningRate(lr=alpha_policy, no_of_parameter_grps=len(list(self.policy.parameters())))
        self.v_hat_lr_adapter = AdaptLearningRate(lr=alpha_value, no_of_parameter_grps=len(list(self.v_hat.parameters())))
        self.alpha_policy = alpha_policy
        self.alpha_value = alpha_value


    def get_action(self, s):
        # Pass the state through the policy neural network, get the probability of each action and sample from it
        x = torch.tensor(np.array([s]), dtype=torch.float32).to(self.device)
        actions = torch.distributions.Categorical(probs=self.policy.forward(x))
        action = actions.sample()
        return action.detach().cpu().item(), actions.log_prob(action)


    def run_episode(self, env):
        # Run one full episode of the environment using the current policy
        trajectory = []
        # Sample initial state
        s, _ = env.reset()
        while True:
            # Get action based on policy and softmax
            a, _ = self.get_action(s)
            # Get next state and reward
            s_next, r, terminated, truncated, info = env.step(a)
            trajectory.append(s)
            trajectory.append(a)
            trajectory.append(r)
            # If terminated by reaching goal state or truncated by reaching max steps of the environment, episode ends
            if terminated or truncated:
                trajectory.append(s_next)
                break
            s = s_next
        return trajectory


    def compute_return(self, trajectory, gamma, t):
        # Compute discounted return from time step t to the end of the episode
        step = 0
        discounted_return = 0
        for k in range(t, len(trajectory) - 1, 3):
            discounted_return = discounted_return + pow(gamma, step) * trajectory[k + 2]
            step = step + 1
        return discounted_return


    def learn_policy(self, env, gamma):

        episodes_return_curve = []
        action_episodescompleted_curve = []

        # Execute 1000 episodes to learn the policy
        for iterations in range(1000):
            
            # Run one full epiosde of the enviornment using current policy
            trajectory = self.run_episode(env)

            # Get the number of actions taken in the trajectory
            action_count = int((len(trajectory) - 1) / 3)
            action_episodescompleted_curve.extend([iterations] * action_count)
            
            # Loop through the time steps in the trajectory
            steps = 0
            
            for i, t in enumerate(range(0, len(trajectory) - 1, 3)):
                
                # For current time step get the state, action, and reward
                s = trajectory[t+0]
                a = trajectory[t+1]
                r = trajectory[t+2]
                
                # Compute return from the current time step as an observed sample of the value function of the current state
                G = self.compute_return(trajectory, gamma, t)
                
                # Get what is the current estimate of the value function of the current state in our neural network
                state_val = self.v_hat.forward(torch.tensor(np.array([s]), dtype=torch.float32).to(self.device))
                
                # Compute the delta, but in no grad so that it's graidient is not computed and it acts like a constant
                with torch.no_grad():
                    delta = G - state_val.detach()

                # Compute the loss which is delta * gradient of the state value function
                # Since optimizer performs gradient descent and we want gradient ascent, we multiply it by -1 
                val_loss = -1 * delta * state_val
                # Clear any graident in the weights of the neural network
                for p in self.v_hat.parameters():
                    p.grad = None
                # Compute gradient of the gradient of the state value function * delta (which act as a constant)
                gradients_v_hat_params = torch.autograd.grad(val_loss, self.v_hat.parameters())
                # Perform gradient ascent step, i.e., 
                # update weights with current adaptive learning rate * delta * graident in the weights of the neural network
                self.v_hat_lr_adapter.update_param(self.v_hat.parameters(), gradients_v_hat_params)
                
                # To compute gradient of the action selected wrt to the wights of the policy network, 
                # we pass the state to the policy network to get the action probabilities
                x = torch.tensor(np.array([s]), dtype=torch.float32).to(self.device)
                actions = torch.distributions.Categorical(probs=self.policy.forward(x))
                # Get the log of the probability of the selected action
                policy_loss = actions.log_prob(torch.tensor(a).to(self.device))
                
                # Compute the loss which is delta * gamma^t * gradient of the policy network
                # Since optimizer performs gradient descent and we want gradient ascent, we multiply it by -1 
                policy_loss = -1 * np.power(gamma, i) * delta * policy_loss
                # Clear any graident in the weights of the neural network
                for p in self.policy.parameters():
                    p.grad = None
                # Compute gradient of the gradient of the policy network * delta (which act as a constant) * gamma^t (which act as a constant)
                gradients_policy_params = torch.autograd.grad(policy_loss, self.policy.parameters())
                # Perform gradient ascent step, i.e., 
                # update weights with current adaptive learning rate * delta * gamma^t * graident in the weights of the neural network
                self.policy_lr_adapter.update_param(self.policy.parameters(), gradients_policy_params)

                steps = steps + 1

            # For debugging print
            if (iterations + 1) % 100 == 0 or iterations == 0:
                print(self.compute_return(trajectory, gamma, 0))

            # Return after episode i, synonym to no of steps needed to reach goal after episode i
            episodes_return_curve.append(self.compute_return(trajectory, gamma, 0))

        return episodes_return_curve, action_episodescompleted_curve
