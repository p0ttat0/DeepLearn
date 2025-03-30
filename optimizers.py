import numpy as np


class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.step = 1
        self.epsilon = 1e-8
        self.b1 = b1
        self.b2 = b2
        self.fme = {}  # First moment estimate
        self.sme = {}  # Second moment estimate

    def adjust_lr(self, layer, weight_gradient, bias_gradient, learning_rate):
        if layer not in self.fme:
            # Initialize moments for this layer
            self.fme[layer] = [np.zeros_like(weight_gradient), np.zeros_like(bias_gradient)]
            self.sme[layer] = [np.zeros_like(weight_gradient), np.zeros_like(bias_gradient)]

        # Update moments first
        self.fme[layer][0] = self.b1 * self.fme[layer][0] + (1 - self.b1) * weight_gradient
        self.fme[layer][1] = self.b1 * self.fme[layer][1] + (1 - self.b1) * bias_gradient
        self.sme[layer][0] = self.b2 * self.sme[layer][0] + (1 - self.b2) * np.square(weight_gradient)
        self.sme[layer][1] = self.b2 * self.sme[layer][1] + (1 - self.b2) * np.square(bias_gradient)

        # Bias correction
        bc_m_weights = self.fme[layer][0] / (1 - self.b1 ** self.step)
        bc_v_weights = self.sme[layer][0] / (1 - self.b2 ** self.step)
        bc_m_bias = self.fme[layer][1] / (1 - self.b1 ** self.step)
        bc_v_bias = self.sme[layer][1] / (1 - self.b2 ** self.step)

        # Compute adaptive learning rates
        weight_change = learning_rate * bc_m_weights / (np.sqrt(bc_v_weights + self.epsilon))
        bias_change = learning_rate * bc_m_bias / (np.sqrt(bc_v_bias + self.epsilon))

        return weight_change, bias_change


class NoOptimizer:
    @staticmethod
    def adjust_lr(layer, weight_gradient, bias_gradient, learning_rate):
        return weight_gradient*learning_rate, bias_gradient*learning_rate
