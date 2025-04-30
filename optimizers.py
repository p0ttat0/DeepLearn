import numpy as np


class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.step = 1
        self.epsilon = 1e-8
        self.b1 = b1
        self.b2 = b2
        self.fme = {}  # First moment estimate
        self.sme = {}  # Second moment estimate

    def adjust_gradient(self, layer, weight_gradient, bias_gradient, learning_rate, dtype=np.float32):
        # Update moments first
        self.fme[layer][0] = (self.b1 * self.fme[layer][0] + (1 - self.b1) * weight_gradient).astype(dtype)
        self.fme[layer][1] = (self.b1 * self.fme[layer][1] + (1 - self.b1) * bias_gradient).astype(dtype)
        self.sme[layer][0] = (self.b2 * self.sme[layer][0] + (1 - self.b2) * np.square(weight_gradient)).astype(dtype)
        self.sme[layer][1] = (self.b2 * self.sme[layer][1] + (1 - self.b2) * np.square(bias_gradient)).astype(dtype)

        # Bias correction to account for initial zero estimates
        bc_m_weights = self.fme[layer][0] / (1 - self.b1 ** self.step)
        bc_v_weights = self.sme[layer][0] / (1 - self.b2 ** self.step)
        bc_m_bias = self.fme[layer][1] / (1 - self.b1 ** self.step)
        bc_v_bias = self.sme[layer][1] / (1 - self.b2 ** self.step)

        # Compute adaptive learning rates
        weight_change = learning_rate * bc_m_weights / (np.sqrt(bc_v_weights + self.epsilon))
        bias_change = learning_rate * bc_m_bias / (np.sqrt(bc_v_bias + self.epsilon))

        return weight_change.astype(dtype), bias_change.astype(dtype)


class NoOptimizer:
    def __init__(self):
        self.step = 1

    @staticmethod
    def adjust_gradient(layer, weight_gradient, bias_gradient, learning_rate):
        return weight_gradient*learning_rate, bias_gradient*learning_rate
