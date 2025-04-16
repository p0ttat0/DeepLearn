import numpy as np
from activationFunctions import ActivationFunction
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange


class Convolution:
    """ input_tensor  : (batchsize, height, width, input_channels)
        kernel : (kernel_height, kernel_width, input_channels, output_channels)
        bias   : (output_channels,)
    """
    def __init__(self, kernel_size: list, activation_function='relu', kernel_initialization='He', bias_initialization='none'):
        valid_activations = ['relu', 'sigmoid', 'tanh', 'swish', 'mish']
        if activation_function not in valid_activations:
            raise ValueError(f"Invalid activation function. Must be one of {valid_activations}")

        self.type = 'convolutional'
        self.kernel_size = kernel_size
        self.input_shape = None
        self.output_shape = None

        self.weights = None
        self.bias = None

        self.kernel_initialization = kernel_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function

        self._ACTIVATION_MAP = {
            'relu': (ActivationFunction.relu, ActivationFunction.d_relu),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.d_sigmoid),
            'tanh': (ActivationFunction.tanh, ActivationFunction.d_tanh),
            'swish': (ActivationFunction.swish, ActivationFunction.d_swish),
            'mish': (ActivationFunction.mish, ActivationFunction.d_mish),
        }

        # back prop
        self.layer_num = None
        self.input_cache = None
        self.unactivated_output_cache = None
        self.kernel_change_cache = None
        self.bias_change_cache = None

        # metrics tracking
        self.output_gradient_magnitude = []
        self.output_gradient_extremes = []
        self.activation_magnitude = []
        self.activation_extremes = []

    def build(self, input_shape: list):
        assert len(input_shape) < 3
        self.input_shape = input_shape

        if self.weights is None:
            self.weights = self.initialize_kernel(self.kernel_initialization, input_shape[3]*self.kernel_size[0]*self.kernel_size[1], self.kernel_size[3]*self.kernel_size[0]*self.kernel_size[1], self.kernel_size)
        if self.bias is None:
            self.bias = self.initialize_bias(self.bias_initialization)

        self.kernel_change_cache = np.zeros(shape=self.weights.shape)
        self.bias_change_cache = np.zeros(shape=self.bias.shape)

    @staticmethod
    def initialize_kernel(initialization: str, input_size: int, output_size: int, kernel_size: list):
        match initialization:
            case 'He' | 'Kaiming':
                return np.random.normal(0, scale=np.sqrt(2.0 / input_size), size=kernel_size)
            case 'Xavier' | 'Glorot':
                return np.random.normal(0, scale=np.sqrt(2.0 / (input_size + output_size)), size=kernel_size)
            case 'LeCun':
                return np.random.normal(0, scale=np.sqrt(1.0 / input_size), size=kernel_size)
            case 'swish':
                return np.random.normal(0, scale=1.1 / np.sqrt(input_size), size=kernel_size)
            case _:
                raise Exception("initialization method not found or not implemented. Maybe check spelling?")

    @staticmethod
    def initialize_bias(initialization: str):
        match initialization:
            case 'none':
                return 0

    @staticmethod
    def conv2d_gemm(input_tensor, kernel, bias, stride=1, padding='same', dtype=np.float32):
        @njit(parallel=True, fastmath=True, cache=True)
        def pad(x, pad_h, pad_w):   # NHWC padding
            batch, in_h, in_w, channels = x.shape
            padded = np.zeros((batch, in_h + 2 * pad_h, in_w + 2 * pad_w, channels), dtype=x.dtype)

            for b in prange(batch):
                for c in prange(channels):
                    padded[b, pad_h:pad_h + in_h, pad_w:pad_w + in_w, c] = x[b, :, :, c]
            return padded
        """
        Args:
            input_tensor  : (batchsize, height, width, input_channels)
            kernel : (kernel_height, kernel_width, input_channels, output_channels)
            bias   : (output_channels,)
            stride : Stride for height/width
            padding: 'same' or 'valid'
        Returns:
            output : (batchsize, output_height, output_width, output_channels)
        """
        # --- Dimensions ---
        batchsize, height, width, input_channels = input_tensor.shape
        kernel_height, kernel_width, _, output_channels = kernel.shape
        stride = (stride, stride) if isinstance(stride, int) else stride

        # --- Padding ---
        if padding == 'same':
            padding_height = (kernel_height - 1) // 2
            padding_width = (kernel_width - 1) // 2
        else:
            padding_height = 0
            padding_width = 0

        padded_input = pad(input_tensor, padding_height, padding_width)
        height_pad, width_pad = padded_input.shape[1], padded_input.shape[2]

        # --- Output Dimensions ---
        output_height = (height_pad - kernel_height) // stride[0] + 1
        output_width = (width_pad - kernel_width) // stride[1] + 1

        batch_stride, height_stride, width_stride, channel_stride = padded_input.strides
        strides = (
            batch_stride,
            height_stride * stride[0],
            width_stride * stride[1],
            height_stride,
            width_stride,
            channel_stride
        )

        windows = as_strided(
            padded_input,
            shape=(batchsize, output_height, output_width, kernel_height, kernel_width, input_channels),
            strides=strides,
            writeable=False
        )

        # --- GEMM Preparation ---  
        x_col = np.reshape(windows, (batchsize * output_height * output_width, kernel_height * kernel_width * input_channels), order='C')
        w_col = np.reshape(kernel, (kernel_height * kernel_width * input_channels, output_channels), order='F')  # Fortran-order for BLAS

        # --- BLAS-Accelerated GEMM ---
        output = np.dot(x_col, w_col).astype(dtype, copy=False)
        output += bias.reshape(1, -1)

        # --- Final Reshape ---
        return output.reshape(batchsize, output_height, output_width, output_channels)


class Dense:
    def __init__(self, size: int, activation_function='relu', weight_initialization='He', bias_initialization='none'):
        """ input_tensor  : (batchsize, input_size)
            weights : (input_size, output_size)
            bias   : (output_size)
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer")

        valid_activations = ['relu', 'sigmoid', 'tanh', 'swish', 'mish', 'softmax']
        if activation_function not in valid_activations:
            raise ValueError(f"Invalid activation function. Must be one of {valid_activations}")

        self.type = 'dense'
        self.size = size
        self.input_shape = None
        self.output_shape = [-1, size]

        self.weights = None
        self.bias = None

        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function

        self._ACTIVATION_MAP = {
            'relu': (ActivationFunction.relu, ActivationFunction.d_relu),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.d_sigmoid),
            'tanh': (ActivationFunction.tanh, ActivationFunction.d_tanh),
            'swish': (ActivationFunction.swish, ActivationFunction.d_swish),
            'mish': (ActivationFunction.mish, ActivationFunction.d_mish),
            'softmax': (ActivationFunction.softmax, ActivationFunction.d_softmax),
        }

        # back prop
        self.layer_num = None
        self.input_cache = None
        self.unactivated_output_cache = None
        self.weight_change_cache = None
        self.bias_change_cache = None

        # metrics tracking
        self.output_gradient_magnitude = []
        self.output_gradient_extremes = []
        self.activation_magnitude = []
        self.activation_extremes = []

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
            case 'swish':
                return np.random.normal(0, scale=1.1 / np.sqrt(input_size), size=(input_size, output_size))
            case _:
                raise Exception("initialization method not found or not implemented. Maybe check spelling?")

    @staticmethod
    def initialize_bias(initialization: str, output_size: int):
        match initialization:
            case 'none':
                return np.zeros((1, output_size))

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0

        unactivated = np.dot(input_tensor, self.weights) + self.bias
        activated = self.get_activation_function()(unactivated)
        self.input_cache = input_tensor
        self.unactivated_output_cache = unactivated

        return activated

    def backprop(self, output_gradient: np.ndarray):
        assert output_gradient.size != 0
        assert self.input_cache is not None
        assert self.unactivated_output_cache is not None

        # --- Partial Derivatives ---
        dz = output_gradient * self.get_d_activation_function()(self.unactivated_output_cache)
        dw = np.dot(self.input_cache.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        di = np.dot(dz, self.weights.T)

        #  --- Gradient Accumulation ---
        self.weight_change_cache += dw
        self.bias_change_cache += db

        # --- Metrics Tracking ---
        self.activation_magnitude = np.mean(np.abs(self.unactivated_output_cache))
        self.activation_extremes = np.max(self.unactivated_output_cache) + np.abs(np.min(self.unactivated_output_cache)) / 2
        self.output_gradient_magnitude = np.mean(np.abs(output_gradient))
        self.output_gradient_extremes = np.max(output_gradient) + np.abs(np.min(output_gradient)) / 2

        return di

    def apply_changes(self, batch_size: int, lr: float, optimizer, clip_value: float):
        assert np.any(self.weight_change_cache)
        assert np.any(self.bias_change_cache)

        #  --- Weights And Biases Update ---
        weight_change, bias_change = optimizer.adjust_gradient(self.layer_num, self.weight_change_cache/batch_size, self.bias_change_cache/batch_size, lr)
        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        # --- Reset Gradient Caches ---
        self.weight_change_cache = np.zeros_like(self.weights)
        self.bias_change_cache = np.zeros_like(self.bias)


class Reshape:
    def __init__(self, output_shape: list):
        self.type = 'reshape'
        self.input_shape = None
        self.output_shape = output_shape
        self.input_cache = None
        self.unactivated_output_cache = None

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0

        self.input_cache = input_tensor
        self.input_shape = input_tensor.shape
        self.unactivated_output_cache = input_tensor.reshape(self.output_shape)
        return self.unactivated_output_cache

    def backprop(self, output_gradient: np.ndarray):
        assert output_gradient.size != 0
        return output_gradient.reshape(self.input_shape)


class Dropout:
    def __init__(self, dropout_rate: float):
        self.type = 'dropout'
        self.dropout_rate = dropout_rate

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0

        binary_tensor = np.random.rand(*input_tensor.shape[1:]) <= (1 - self.dropout_rate)
        return input_tensor*binary_tensor/(1 - self.dropout_rate)

    @staticmethod
    def backprop(output_gradient: np.ndarray):
        assert output_gradient.size != 0
        return output_gradient
