import torch

class AdaptLearningRate():

    def __init__(self, lr, no_of_parameter_grps):
        self.m = [0] * no_of_parameter_grps
        self.v = [0] * no_of_parameter_grps
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 1
        self.lr = lr

    def update_param(self, params, gradient):
        # Updating parameters based on Adam optimizer policy
        for i, p in enumerate(params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.pow(gradient[i], 2)
            m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t))
            adapted_gradient = self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            p.data = p.data - adapted_gradient
        self.t = self.t + 1