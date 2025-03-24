import numpy as np


class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.step = 0
        self.epsilon = 10 ** -8
        self.b1 = b1
        self.b2 = b2
        self.fme = {}  # first moment estimate
        self.sme = {}  # second moment estimate
        self.prev_fme = {}  # first moment estimate
        self.prev_sme = {}  # second moment estimate

    def adjust_lr(self, layer, weight_gradient, bias_gradient, learning_rate):
        self.step += 1

        m_weights = self.b1 * self.prev_fme[layer][0] + (1 - self.b1) * weight_gradient
        m_bias = self.b1 * self.prev_fme[layer][1] + (1 - self.b1) * bias_gradient

        v_weights = self.b2 * self.prev_sme[layer][0] + (1 - self.b2) * np.square(weight_gradient)
        v_bias = self.b2 * self.prev_sme[layer][1] + (1 - self.b2) * np.square(bias_gradient)

        self.prev_fme[layer][1], self.prev_fme[layer][0] = self.fme[layer][1], self.fme[layer][0]
        self.prev_sme[layer][1], self.prev_sme[layer][0] = self.sme[layer][1], self.sme[layer][0]
        self.fme[layer][1], self.fme[layer][0] = m_bias, m_weights
        self.sme[layer][1], self.sme[layer][0] = v_bias, v_weights

        bc_m_weights = m_weights / (1 - self.b1 ** self.step)
        bc_v_weights = v_weights / (1 - self.b1 ** self.step)

        bc_m_bias = m_bias / (1 - self.b1 ** self.step)
        bc_v_bias = v_bias / (1 - self.b1 ** self.step)

        weight_lr = bc_m_weights / (np.sqrt(bc_v_weights) + self.epsilon) * learning_rate
        bias_lr = bc_m_bias / (np.sqrt(bc_v_bias) + self.epsilon) * learning_rate

        return weight_lr, bias_lr

class NoOptimizer:
    @staticmethod
    def adjust_lr(layer, weight_gradient, bias_gradient, learning_rate):
        return learning_rate, learning_rate
