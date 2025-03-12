import numba
import numpy as np
import math


class Dense:
    def __init__(self, size: int(), activation_function='ReLU', weight_initialization='He', bias_initialization='none'):
        self.type = 'dense'
        self.size = size
        self.output_size = size
        self.weights = None
        self.bias = None
        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function

    def build(self, input_size):
        self.weights = self.initialize_weights(self.weight_initialization, input_size, self.size)
        self.bias = self.initialize_bias(self.bias_initialization, self.size)

    @numba.jit(locals={'x': numpy}, nopython=True)
    @staticmethod
    def act_function(activation_function, x):
        match activation_function:
            case 'ReLU':
                return x * (x > 0)
            case 'sigmoid':
                return 1 / (1 + np.exp(-x))
            case 'tanh':
                return np.tanh(x)
            case 'swish':
                return x / (1 + np.exp(-x))
            case 'mish':
                return x * np.tanh(np.log(1 + np.exp(x)))
            case 'softmax':
                return np.exp(x) / np.sum(np.exp(x), axis=0)
            case _:
                raise Exception("activation function not found or not implemented. Maybe check spelling?")

    @staticmethod
    def d_act_function(activation_function, x):
        match activation_function:
            case 'ReLU':
                return (x > 0).astype(float)
            case 'sigmoid':
                return 1 / (1 + np.exp(-x))
            case 'tanh':
                return np.tanh(x)
            case 'swish':
                return x / (1 + np.exp(-x))
            case 'mish':
                return x * np.tanh(np.log(1 + np.exp(x)))
            case 'softmax':
                return np.exp(x) / np.sum(np.exp(x), axis=0)
            case _:
                raise Exception("activation function not found or not implemented. Maybe check spelling?")

    @staticmethod
    def initialize_weights(initialization, input_size: int(), output_size: int()):
        match initialization:
            case 'He' | 'Kaiming':
                return np.random.normal(0, math.sqrt(2 / input_size), (output_size, input_size))
            case 'Xavier' | 'Glorot':
                return np.random.normal(0, math.sqrt(2 / input_size + output_size), (output_size, input_size))
            case 'LeCun':
                return np.random.normal(0, math.sqrt(1 / input_size), (output_size, input_size))
            case _:
                raise Exception("initialization method not found or not implemented. Maybe check spelling?")

    @staticmethod
    def initialize_bias(initialization, output_size: int()):
        match initialization:
            case 'none':
                return np.zeros((output_size, 1))

    def forward(self, x):
        return self.act_function(self.activation_function, np.dot(self.weights, x) + self.bias)

    def back_prop(self, loss):
        unactivated_loss = self.d_act_function(loss)
