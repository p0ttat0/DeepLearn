import numba as nb
import numpy as np
from activationFunctions import ActivationFunction


class Dense:
    def __init__(self, size: int, activation_function='relu', weight_initialization='He', bias_initialization='none'):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer")

        valid_activations = ['relu', 'sigmoid', 'tanh', 'swish', 'mish', 'softmax']
        if activation_function not in valid_activations:
            raise ValueError(f"Invalid activation function. Must be one of {valid_activations}")

        self._ACTIVATION_MAP = {
            'relu': (ActivationFunction.relu, ActivationFunction.d_relu),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.d_sigmoid),
            'tanh': (ActivationFunction.tanh, ActivationFunction.d_tanh),
            'swish': (ActivationFunction.swish, ActivationFunction.d_swish),
            'mish': (ActivationFunction.mish, ActivationFunction.d_mish),
            'softmax': (ActivationFunction.softmax, ActivationFunction.d_softmax),
        }

        self.type = 'dense'
        self.size = size
        self.input_shape = None
        self.output_shape = [-1, size]
        self.layer_num = None

        self.weights = None
        self.bias = None

        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function

        self.input_cache = None
        self.unactivated_output_cache = None
        self.weight_change_cache = None
        self.bias_change_cache = None

    def build(self, input_shape: list):
        assert len(input_shape) < 3
        self.input_shape = input_shape

        if self.weights is None:
            self.weights = self.initialize_weights(self.weight_initialization, input_shape[1], self.size)
        if self.bias is None:
            self.bias = self.initialize_bias(self.bias_initialization, self.size)

        self.weight_change_cache = np.zeros(shape=self.weights.shape)
        self.bias_change_cache = np.zeros(shape=self.bias.shape)

    def get_activation_function(self):
        return self._ACTIVATION_MAP[self.activation_function][0]

    def get_d_activation_function(self):
        return self._ACTIVATION_MAP[self.activation_function][1]

    @staticmethod
    def initialize_weights(initialization: str, input_size: int, output_size: int):
        match initialization:
            case 'He' | 'Kaiming':
                return np.random.normal(0, scale=np.sqrt(2.0 / input_size), size=(input_size, output_size))
            case 'Xavier' | 'Glorot':
                return np.random.normal(0, scale=np.sqrt(2.0 / (input_size + output_size)), size=(input_size, output_size))
            case 'LeCun':
                return np.random.normal(0, scale=np.sqrt(1.0 / input_size), size=(input_size, output_size))
            case _:
                raise Exception("initialization method not found or not implemented. Maybe check spelling?")

    @staticmethod
    def initialize_bias(initialization: str, output_size: int):
        match initialization:
            case 'none':
                return np.zeros((1, output_size))

    def forprop(self, x: np.ndarray):
        unactivated = np.dot(x, self.weights) + self.bias
        activated = self.get_activation_function()(unactivated)
        self.input_cache = x
        self.unactivated_output_cache = unactivated
        return activated

    def backprop(self, output_gradient: np.ndarray):
        assert self.input_cache is not None
        assert self.unactivated_output_cache is not None

        dz = output_gradient * self.get_d_activation_function()(self.unactivated_output_cache)
        dw = np.dot(self.input_cache.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        di = np.dot(dz, self.weights.T)

        # Accumulate gradients
        self.weight_change_cache += dw
        self.bias_change_cache += db

        return di

    def apply_changes(self, batch_size: int, lr: float, optimizer, clip_value: float):
        # Update weights and biases
        self.weight_change_cache /= batch_size
        self.bias_change_cache /= batch_size

        weight_change, bias_change = optimizer.adjust_lr(self.layer_num, self.weight_change_cache, self.bias_change_cache, lr)

        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        # print(f"Layer {self.layer_num}: Weight grad mean={np.mean(self.weight_change_cache)}, Bias grad mean={np.mean(self.bias_change_cache)}")

        # Reset gradient caches
        self.weight_change_cache = np.zeros_like(self.weights)
        self.bias_change_cache = np.zeros_like(self.bias)


class Reshape:
    def __init__(self, input_shape: list, output_shape: list):
        self.type = 'reshape'
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_cache = None
        self.unactivated_output_cache = None

    def forprop(self, x):
        self.input_cache = x
        self.unactivated_output_cache = x.reshape(self.output_shape)
        return self.unactivated_output_cache

    def backprop(self, x):
        return x.reshape(self.input_shape)
