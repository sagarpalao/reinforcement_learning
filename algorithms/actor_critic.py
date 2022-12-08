import torch
import torch.nn as nn
from torch import optim
import numpy as np

# A custom layer to enable exploration using Boltzmann distribution
class ExplorationLayer(nn.Module):

    def __init__(self, exploration):
        super().__init__()
        self.exploration = exploration

    def forward(self, x):
        # Multiply exploration constant to x
        # exploration -> 0, more and more stochastic action selection
        # exploration -> inf, more and more deterministic action selection
        return self.exploration * x


# Multilayer Neural Network with ReLU activation to represent Policy
class PolicyNeuralNetwork(nn.Module):

    def __init__(self, no_of_states, hidden_units, no_of_actions, exploration):
        super(PolicyNeuralNetwork, self).__init__()
        layers = []
        for i, k in enumerate(hidden_units):
            layers.append(nn.Linear(no_of_states if i == 0 else hidden_units[i - 1], k))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], no_of_actions))
        layers.append(ExplorationLayer(exploration))
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


# Actro Critic Method
class ActorCritic():

    def __init__(self, no_of_states, hidden_units, no_of_actions, alpha_policy, alpha_value, exploration=1):
        # Create neural network to represent policy and state-value function
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = PolicyNeuralNetwork(no_of_states, hidden_units, no_of_actions, exploration).to(self.device)
        self.v_hat = ValueFunctionNeuralNetwork(no_of_states, hidden_units).to(self.device)
        # Construct optimizers to optimize weights of the neural network to represent policy and state-value function
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=alpha_policy)
        self.stateval_optimizer = optim.Adam(self.v_hat.parameters(), lr=alpha_value)


    def get_action(self, s):
        # Pass the state through the policy neural network, get the probability of each action and sample from it
        x = torch.tensor(np.array([s]), dtype=torch.float32).to(self.device)
        actions = torch.distributions.Categorical(probs=self.policy.forward(x))
        action = actions.sample()
        return action.detach().cpu().item(), actions.log_prob(action)


    def learn_policy(self, env, gamma):
        
        episodes_return_curve = []
        action_episodescompleted_curve = []

        # Execute 1000 episodes to learn the policy
        for iterations in range(1000):

            # Sample initial state
            s, _ = env.reset()
            I = 1
            steps = 0
            
            G = 0
            while True:
                # Get action based on policy and softmax and the log of the probability of action of its computation graph
                # to compute the gradient
                a, policy_loss = self.get_action(s)
                # Get next state and reward
                s_next, r, terminated, truncated, info = env.step(a)
                action_episodescompleted_curve.append(iterations)

                # Compute the running return of the episode for the lurning curve
                G = G + np.power(gamma, steps) * r
                
                # Get what is the current estimate of the value function of the current state in our neural network
                state_val = self.v_hat.forward(torch.tensor(np.array([s]), dtype=torch.float32).to(self.device))

                # Compute the delta, but in no grad so that it's graidient is not computed and it acts like a constant
                with torch.no_grad():
                    new_state_val = self.v_hat.forward(torch.tensor(np.array([s_next]), dtype=torch.float32).to(self.device)).detach()
                    if terminated:
                        delta = r - state_val.detach()
                    else:
                        delta = r + gamma * new_state_val - state_val.detach()

                # Compute the loss which is delta * gradient of the state value function
                # Since optimizer performs gradient descent and we want gradient ascent, we multiply it by -1 
                val_loss = -1 * delta * state_val
                # Clear any graident in the weights of the neural network
                self.stateval_optimizer.zero_grad()
                # Compute gradient of the gradient of the state value function * delta (which act as a constant)
                val_loss.backward()
                # Perform gradient ascent step, i.e., 
                # update weights with current adaptive learning rate * delta * graident in the weights of the neural network
                self.stateval_optimizer.step()

                # Compute the loss which is delta * I * gradient of the policy network
                # Since optimizer performs gradient descent and we want gradient ascent, we multiply it by -1 
                policy_loss = -1 * delta * I * policy_loss
                # Clear any graident in the weights of the neural network
                self.policy_optimizer.zero_grad()
                # Compute gradient of the gradient of the policy network * delta (which act as a constant) * I (which act as a constant)
                policy_loss.backward()
                # Perform gradient ascent step, i.e., 
                # update weights with current adaptive learning rate * delta * gamma^t * graident in the weights of the neural network
                self.policy_optimizer.step()

                # update I 
                I = I * gamma

                # Update step counter, i.e. no of actions taken so far in the episode
                steps = steps + 1

                # If terminated by reaching goal state or truncated by reaching max steps of the environment, episode ends
                if terminated or truncated:
                    break
                
                # Repeat with the next state
                s = s_next

            if (iterations + 1) % 100 == 0 or iterations == 0:
                print(G)

            # Return after episode i, synonym to no of steps needed to reach goal after episode i
            episodes_return_curve.append(G)

        return episodes_return_curve, action_episodescompleted_curve






