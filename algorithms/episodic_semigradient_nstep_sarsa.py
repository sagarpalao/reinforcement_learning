import torch
import torch.nn as nn
import random
import numpy as np
from utils.param_update import ParameterUpdate


# Multilayer Neural Network with ReLU activation to represent Action-Value Function
# Action-value function of the state and action given a state and a action
class ValueFunctionNeuralNetwork(nn.Module):

    def __init__(self, no_of_states_actions, hidden_units):
        super(ValueFunctionNeuralNetwork, self).__init__()
        layers = []
        for i, k in enumerate(hidden_units):
            layers.append(nn.Linear(no_of_states_actions if i == 0 else hidden_units[i - 1], k))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.sequence_linear_relu = nn.Sequential(*layers)  

    def forward(self, x):
        out = self.sequence_linear_relu(x)
        return out


# Episodic Semigradient N Step SARSA
class EpisodicSemigradientNStepSarsa():

    def __init__(self, no_of_states, hidden_units, no_of_actions, alpha_value, N, epsilon, decay=False, beta=0):
        # Create neural network to represent action-value function
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_hat = ValueFunctionNeuralNetwork(no_of_states + no_of_actions, hidden_units).to(self.device)
        self.q_hat_param_update = ParameterUpdate(lr=alpha_value, no_of_parameter_grps=len(list(self.q_hat.parameters())))
        self.alpha_value = alpha_value
        self.no_of_actions = no_of_actions
        self.N = N
        self.epsilon = epsilon
        self.decay = decay
        self.beta = beta


    def get_action(self, s, epsilon):
        
        # Pass the state through the action-value neural network, |A| times with |A| different actions 
        # and get q-values for each action of the state
        action_q_vals = []
        for a in range(self.no_of_actions):
            action_one_hot = np.zeros(self.no_of_actions)
            action_one_hot[a] = 1
            input = np.append(s, action_one_hot)
            x = torch.tensor(np.array([input]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_val = self.q_hat.forward(x).detach().cpu().numpy().item()
            action_q_vals.append(q_val)
        action_q_vals = np.array(action_q_vals)

        # choose an action based on epsilon-greedy action selection
        weights = np.zeros(self.no_of_actions)
        a_star = np.argwhere(action_q_vals == np.max(action_q_vals)).flatten().tolist()
        for a in range(self.no_of_actions):
            if a in a_star:
                weights[a] = ((1 - epsilon) / len(a_star)) + (epsilon / self.no_of_actions)
            else:
                weights[a] = epsilon / self.no_of_actions
        return random.choices(range(self.no_of_actions), weights, k=1)[0]


    def compute_return(self, trajectory, gamma, start):
        # Compute discounted return from time step t to the current end of the trajectory representing t
        step = 0
        discounted_return = 0
        for k in range(start, len(trajectory) - 2, 3):
            discounted_return = discounted_return + pow(gamma, step) * trajectory[k + 2]
            step = step + 1
        return discounted_return


    def learn_policy(self, env, gamma):

        episodes_return_curve = []
        action_episodescompleted_curve = []

        # Execute 1000 episodes to learn the policy
        for iterations in range(1000):

            trajectory = []
            # Sample initial state
            s, _ = env.reset()
            # Get action based on epsilon-greedy action selection from q-value
            a = self.get_action(s, self.epsilon)
            trajectory.append(s)
            trajectory.append(a)
            T = np.inf
            t = 0

            while True:
                # If current time step is less than the end of episode, continue running and building trajectory
                if t < T:
                    # Get next state and reward
                    s_next, r, terminated, truncated, info = env.step(a)
                    trajectory.append(r)
                    trajectory.append(s_next)
                    action_episodescompleted_curve.append(iterations)
                    # If terminated by reaching goal state or truncated by reaching max steps of the environment, 
                    # update T to map to the final time step of the trajectory
                    if terminated or truncated:
                        T = t + 1
                    else:
                        # If not terminated, Get action based on epsilon-greedy action selection from q-value
                        a = self.get_action(s_next, self.epsilon)
                        trajectory.append(a)
                # Check if all timestep have been updated, if yes break the learning process
                rho = t - self.N + 1
                if rho == T:
                    break
                if rho >= 0:
                    # Compute the discounted reward from the next N time step
                    G = self.compute_return(trajectory, gamma, rho*3)
                    # If next state exists after N time step, compute its q-value function 
                    if rho + self.N < T:
                        with torch.no_grad():
                            S_rho_plus_N = trajectory[(rho + self.N)*3]
                            A_rho_plus_N = trajectory[(rho + self.N)*3 + 1]
                            action_one_hot = np.zeros(self.no_of_actions)
                            action_one_hot[A_rho_plus_N] = 1
                            input = np.append(S_rho_plus_N, action_one_hot)
                            x = torch.tensor(np.array([input]), dtype=torch.float32).to(self.device)
                            q_val = self.q_hat.forward(x).detach().cpu().numpy().item()
                        # Update estimate of G with discounted reward from the next N time step and 
                        # q-value function of timestep at N + 1, if it exists
                        G = G + pow(gamma, self.N) * q_val
                    
                    # For the state and action to be updated, forward pass through through the action-value neural network 
                    # to compute the gradient
                    S_rho = trajectory[rho*3]
                    A_rho = trajectory[rho*3 + 1]
                    action_one_hot = np.zeros(self.no_of_actions)
                    action_one_hot[A_rho] = 1
                    input = np.append(S_rho, action_one_hot)
                    x = torch.tensor(np.array([input]), dtype=torch.float32).to(self.device)
                    q_val_rho = self.q_hat.forward(x)

                    # Compute delta as a constant
                    delta = G - q_val_rho.detach().cpu().numpy().item()

                    # Compute the loss which is delta * gradient of the action value function
                    # Since we perform gradient descent and we want gradient ascent, we multiply it by -1 
                    loss = -1 * delta * q_val_rho
                    # Clear any graident in the weights of the neural network
                    for p in self.q_hat.parameters():
                        p.grad = None
                    # Compute gradient of the gradient of the action value function * delta (which act as a constant)
                    gradients_q_hat_params = torch.autograd.grad(loss, self.q_hat.parameters())
                    # Perform gradient ascent step, i.e., 
                    # update weights with current adaptive learning rate * delta * graident in the weights of the neural network
                    self.q_hat_param_update.update_param(self.q_hat.parameters(), gradients_q_hat_params)

                t = t + 1

            # If decay, decay the exploration exponentially with beta after every 10 episode runs
            if self.decay and iterations%10==0:
                self.epsilon = self.epsilon * self.beta

            # To debug
            if (iterations + 1) % 100 == 0 or iterations == 0:
                print(self.compute_return(trajectory, gamma, 0))

            # Return after episode i, synonym to no of steps needed to reach goal after episode i
            episodes_return_curve.append(self.compute_return(trajectory, gamma, 0))

        return episodes_return_curve, action_episodescompleted_curve
