import numpy as np
from activationFunctions import ActivationFunction
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange


class Convolution:
    def __init__(self, kernel_size: int, activation_function='relu', kernel_initialization='He', bias_initialization='none'):
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

        self.type = 'convolutional'
        self.size = kernel_size
        self.input_shape = None
        self.output_shape = None

        self.weights = None
        self.bias = None

        self.kernel_initialization = kernel_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function

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

    @staticmethod
    def conv2d_gemm(x, kernel, bias, stride=1, padding='same', dtype=np.float32):
        @njit(parallel=True, fastmath=True, cache=True)
        def pad_input(x, pad_h, pad_w):
            """NHWC padding optimized with Numba. 2.5x faster than numpy.pad."""
            batch, in_h, in_w, channels = x.shape
            padded = np.zeros((batch, in_h + 2 * pad_h, in_w + 2 * pad_w, channels), dtype=x.dtype)

            for b in prange(batch):
                for c in prange(channels):
                    padded[b, pad_h:pad_h + in_h, pad_w:pad_w + in_w, c] = x[b, :, :, c]
            return padded
        """
        Args:
            x  : (batchsize, height, width, input_channels)
            kernel : (kernel_height, kernel_width, input_channels, output_channels)
            bias   : (output_channels,)
            stride : Stride for height/width
            padding: 'same' or 'valid'
            dtype  : Output dtype (float32 recommended)

        Returns:
            output : (batchsize, output_height, output_width, output_channels)
        """
        # --- Pre-checks ---
        x = np.ascontiguousarray(x)
        kernel = np.ascontiguousarray(kernel)
        bias = np.ascontiguousarray(bias)

        # --- Dimensions ---
        batchsize, height, width, input_channels = x.shape
        kernel_height, kernel_width, _, output_channels = kernel.shape
        stride = (stride, stride) if isinstance(stride, int) else stride

        # --- Padding ---
        if padding == 'same':
            padding_height = (kernel_height - 1) // 2
            padding_width = (kernel_width - 1) // 2
        else:
            padding_height = padding_width = 0

        padded_input = pad_input(x, padding_height, padding_width)
        height_pad, width_pad = padded_input.shape[1], padded_input.shape[2]

        # --- Output Dimensions ---
        output_height = (height_pad - kernel_height) // stride[0] + 1
        output_width = (width_pad - kernel_width) // stride[1] + 1

        # --- as_strided Magic ---
        # Key Insight: Compute strides manually to avoid numpy's generic stride calc
        s_b, s_h, s_w, s_c = padded_input.strides
        strides = (
            s_b,  # Batch stride
            s_h * stride[0],  # height stride (jump by stride)
            s_w * stride[1],  # width stride (jump by stride)
            s_h,  # kernel_height stride
            s_w,  # kernel_width stride
            s_c  # Channel stride
        )

        # Shape: (batchsize, output_height, output_width, kernel_height, kernel_width, input_channels)
        windows = as_strided(
            padded_input,
            shape=(batchsize, output_height, output_width, kernel_height, kernel_width, input_channels),
            strides=strides,
            writeable=False
        )

        # --- GEMM Preparation ---
        # Reshape to (batchsize*output_height*output_width, kernel_height*kernel_width*input_channels) without copying
        # Force C-contiguous for BLAS (critical for performance)
        x_col = np.reshape(windows, (batchsize * output_height * output_width, kernel_height * kernel_width * input_channels), order='C')
        w_col = np.reshape(kernel, (kernel_height * kernel_width * input_channels, output_channels), order='F')  # Fortran-order for BLAS

        # --- BLAS-Accelerated GEMM ---
        # Equivalent to: output = (x_col @ w_col) + bias
        # Use np.dot with contiguous arrays for MKL/OpenBLAS acceleration
        output = np.dot(x_col, w_col).astype(dtype, copy=False)
        output += bias.reshape(1, -1)  # In-place broadcasted add

        # --- Final Reshape ---
        return output.reshape(batchsize, output_height, output_width, output_channels)


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

        self.weights = None
        self.bias = None

        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function

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

        # metrics tracking
        self.activation_magnitude = np.mean(np.abs(self.unactivated_output_cache))
        self.activation_extremes = np.max(self.unactivated_output_cache) + np.abs(np.min(self.unactivated_output_cache)) / 2
        self.output_gradient_magnitude = np.mean(np.abs(output_gradient))
        self.output_gradient_extremes = np.max(output_gradient) + np.abs(np.min(output_gradient)) / 2

        return di

    def apply_changes(self, batch_size: int, lr: float, optimizer, clip_value: float):
        # Update weights and biases
        self.weight_change_cache /= batch_size
        self.bias_change_cache /= batch_size

        weight_change, bias_change = optimizer.adjust_gradient(self.layer_num, self.weight_change_cache, self.bias_change_cache, lr)

        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

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


class Dropout:
    def __init__(self, dropout_rate):
        self.type = 'dropout'
        self.dropout_rate = dropout_rate

    def forprop(self, x):
        binary_matrix = np.random.rand(*x.shape[1:]) <= (1 - self.dropout_rate)
        return x*binary_matrix/(1 - self.dropout_rate)

    @staticmethod
    def backprop(output_gradient):
        return output_gradient
