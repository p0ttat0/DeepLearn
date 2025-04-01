import numpy as np


class ActivationFunction:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def swish(x):
        return x / (1 + np.exp(-x))

    @staticmethod
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def d_relu(x):
        return (x > 0).astype(np.int8)

    @staticmethod
    def d_sigmoid(x):
        activated = 1 / (1 + np.exp(-x))
        return activated * (1 - activated)

    @staticmethod
    def d_tanh(x):
        activated = np.tanh(x)
        return 1 - np.square(activated)

    @staticmethod
    def d_swish(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 + x * (1 - sig))

    @staticmethod
    def d_mish(x):  # maybe right
        sp = np.log1p(np.exp(x))  # soft plus(x)
        tsp = np.tanh(sp)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return tsp + x * sigmoid * (1.0 - tsp * tsp)

    @staticmethod
    # assumes categorical cross entropy
    def d_softmax(x):
        return np.ones_like(x)
