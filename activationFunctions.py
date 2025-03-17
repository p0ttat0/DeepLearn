import numpy as np
import numba as nb


class ActivationFunction:

    @staticmethod
    @nb.njit(cache=True)
    def relu(x):
        return x * (x > 0)

    @staticmethod
    @nb.njit(cache=True)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    @nb.njit(cache=True)
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    @nb.njit(cache=True)
    def swish(x):
        return x / (1 + np.exp(-x))

    @staticmethod
    @nb.njit(cache=True)
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    @staticmethod
    @nb.njit(cache=True)
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)

    # derivatives assumes x is the output of activation function:
    @staticmethod
    @nb.njit(cache=True)
    def d_relu(x):
        return (x > 0).astype(float)

    @staticmethod
    @nb.njit(cache=True)
    def d_sigmoid(x):
        return x * (1 - x)

    @staticmethod
    @nb.njit(cache=True)
    def d_tanh(x):
        return 1 - np.square(x)

    @staticmethod
    @nb.njit(cache=True)
    def d_swish(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 + x * (1 - sig))

    @staticmethod
    @nb.njit(cache=True)
    def d_mish(x):      # maybe right
        sp = np.log1p(np.exp(x))  # soft plus(x)
        tsp = np.tanh(sp)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return tsp + x * sigmoid * (1.0 - tsp * tsp)

    @staticmethod
    @nb.njit(cache=True)        # assumes categorical cross entropy
    def d_softmax(x):
        return x
