import numba as nb
import numpy as np
from activationFunctions import ActivationFunction


class Dense:
    def __init__(self, size: int(), activation_function='relu', weight_initialization='He', bias_initialization='none'):
        self.type = 'dense'
        self.size = size
        self.input_shape = None
        self.output_shape = [size]
        self.layer_num = None

        self.weights = None
        self.bias = None

        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function
        """        self.activation_functions = {
                    'relu': 0,
                    'sigmoid': 1,
                    'tanh': 2,
                    'swish': 3,
                    'mish': 4,
                    'softmax': 5,
                }
        """
        self.input_cache = None
        self.unactivated_output_cache = None
        self.weight_change_cache = None
        self.bias_change_cache = None

    def build(self, input_shape: list):
        assert len(input_shape) < 3
        self.input_shape = input_shape

        if self.weights is None:
            self.weights = self.initialize_weights(self.weight_initialization, input_shape[0], self.size)
        if self.bias is None:
            self.bias = self.initialize_bias(self.bias_initialization, self.size)

        self.weight_change_cache = np.zeros(shape=self.weights.shape)
        self.bias_change_cache = np.zeros(shape=self.bias.shape)

    def get_activation_function(self):
        match self.activation_function:
            case 'relu':
                return ActivationFunction.relu
            case 'sigmoid':
                return ActivationFunction.sigmoid
            case 'tanh':
                return ActivationFunction.tanh
            case 'swish':
                return ActivationFunction.swish
            case 'mish':
                return ActivationFunction.mish
            case 'softmax':
                return ActivationFunction.softmax
            case _:
                raise Exception(f"{self.activation_function} of type {type(self.activation_function)} not found or not "
                                f"implemented. Maybe check spelling?")

    def get_d_activation_function(self):
        match self.activation_function:
            case 'relu':
                return ActivationFunction.d_relu
            case 'sigmoid':
                return ActivationFunction.d_sigmoid
            case 'tanh':
                return ActivationFunction.d_tanh
            case 'swish':
                return ActivationFunction.d_swish
            case 'mish':
                return ActivationFunction.d_mish
            case 'softmax':
                return ActivationFunction.d_softmax
            case _:
                raise Exception(f"{self.activation_function} of type {type(self.activation_function)} not found or not "
                                f"implemented. Maybe check spelling?")

    @staticmethod
    def initialize_weights(initialization, input_size: int(), output_size: int()):
        match initialization:
            case 'He' | 'Kaiming':
                return np.random.normal(0, np.sqrt(2 / input_size), (output_size, input_size))
            case 'Xavier' | 'Glorot':
                return np.random.normal(0, np.sqrt(2.0 / (input_size + output_size)), size=(output_size, input_size))
            case 'LeCun':
                return np.random.normal(0, np.sqrt(1 / input_size), (output_size, input_size))
            case _:
                raise Exception("initialization method not found or not implemented. Maybe check spelling?")

    @staticmethod
    def initialize_bias(initialization, output_size: int()):
        match initialization:
            case 'none':
                return np.zeros((output_size, 1))

    def forprop(self, x):
        @nb.njit(cache=True)
        def forward(inpt: np.ndarray, weights: np.ndarray, bias: np.ndarray, act_func):
            unactivated = np.dot(weights, inpt) + bias
            activated = act_func(unactivated)
            return unactivated, activated

        activations = forward(x, self.weights, self.bias, self.get_activation_function())
        self.input_cache = x
        self.unactivated_output_cache = activations[0]
        return activations[1]

    def backprop(self, output_gradient, clip_value=1):
        assert self.input_cache is not None
        assert self.unactivated_output_cache is not None

        @nb.njit(cache=True)
        def calculate_gradients(out_gradient, inputs, weights, unactivated_output, d_act_func, clip_amount):
            dz = out_gradient * d_act_func(unactivated_output)
            dw = np.clip(dz.dot(inputs.T), -clip_amount, clip_amount)
            db = np.clip(np.sum(dz, axis=1).reshape(-1, 1), -clip_amount, clip_amount)
            di = np.clip(weights.T.dot(dz), -clip_amount, clip_amount)

            return dw, db, di

        d_w, d_b, d_i = calculate_gradients(output_gradient, self.input_cache, self.weights, self.unactivated_output_cache, self.get_d_activation_function(), clip_value)

        # Accumulate gradients
        self.weight_change_cache += d_w
        self.bias_change_cache += d_b

        return d_i

    def apply_changes(self, batch_size, lr, optimizer):
        # Update weights and biases
        self.weight_change_cache /= batch_size
        self.bias_change_cache /= batch_size

        weight_lr, bias_lr = optimizer.adjust_lr(self.layer_num, self.weight_change_cache, self.bias_change_cache, lr)

        self.weights -= self.weight_change_cache * weight_lr
        self.bias -= self.bias_change_cache * bias_lr

        # Reset gradient caches
        self.weight_change_cache = np.zeros_like(self.weights)
        self.bias_change_cache = np.zeros_like(self.bias)


class Flatten:
    def __init__(self, input_shape: list):
        self.type = 'flatten'
        self.input_shape = input_shape
        if len(input_shape) > 1:
            self.output_shape = [int(np.prod(input_shape[-2:])), -1]
        else:
            self.output_shape = input_shape

    def build(self, input_shape):   # nothing to initialize
        pass

    def apply_changes(self, batch_size, learning_rate, optimizer):    # nothing to change
        pass

    def forprop(self, x):
        return x.reshape(self.output_shape)

    def backprop(self, x, clip_value=None):
        return x.reshape(self.input_shape)
