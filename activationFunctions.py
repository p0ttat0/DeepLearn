import numpy as np
import numba as nb


class ActivationFunction:

    @staticmethod
    ## @nb.njit(cache=True)
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    # @nb.njit(cache=True)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    # @nb.njit(cache=True)
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    # @nb.njit(cache=True)
    def swish(x):
        return x / (1 + np.exp(-x))

    @staticmethod
    # @nb.njit(cache=True)
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    @staticmethod
    # @nb.njit(cache=True)
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    # @nb.njit(cache=True)
    def d_relu(x):
        return (x > 0).astype(np.int8)

    @staticmethod
    # @nb.njit(cache=True)
    def d_sigmoid(x):
        activated = 1 / (1 + np.exp(-x))
        return activated * (1 - activated)

    @staticmethod
    # @nb.njit(cache=True)
    def d_tanh(x):
        activated = np.tanh(x)
        return 1 - np.square(activated)

    @staticmethod
    # @nb.njit(cache=True)
    def d_swish(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 + x * (1 - sig))

    @staticmethod
    # @nb.njit(cache=True)
    def d_mish(x):      # maybe right
        sp = np.log1p(np.exp(x))  # soft plus(x)
        tsp = np.tanh(sp)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return tsp + x * sigmoid * (1.0 - tsp * tsp)

    @staticmethod
    # @nb.njit(cache=True)        # assumes categorical cross entropy
    def d_softmax(x):
        return np.ones_like(x)
