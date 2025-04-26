import numpy as np
from activationFunctions import ActivationFunction
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange


class Convolution:
    """ input_tensor  : (batchsize, height, width, input_channels)
        kernel : (kernel_height, kernel_width, input_channels, output_channels)
        bias   : (output_channels,)
        stride : (height_stride, width_stride)
        padding: (vertical_padding, horizontal_padding)
    """
    def __init__(self, kernel_shape: list, activation_function='relu', weight_initialization='He', bias_initialization='none', padding='valid', stride=1, dtype=np.float32):
        valid_activations = ['relu', 'sigmoid', 'tanh', 'swish']
        if activation_function not in valid_activations:
            raise ValueError(f"Invalid activation function. Must be one of {valid_activations}")

        self._ACTIVATION_MAP = {
            'relu': (ActivationFunction.relu, ActivationFunction.d_relu),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.d_sigmoid),
            'tanh': (ActivationFunction.tanh, ActivationFunction.d_tanh),
            'swish': (ActivationFunction.swish, ActivationFunction.d_swish),
        }

        # --- layer general attributes ---
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.input_shape = None
        self.output_shape = None

        # --- layer type specific attributes ---
        self.kernel = None
        self.bias = None
        self.padding = self.get_padding_obj(padding)
        self.stride = [stride, stride] if isinstance(stride, int) else stride
        self.dtype = dtype

        self.kernel_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function
        self.act_func = self._ACTIVATION_MAP[activation_function][0]
        self.d_act_func = self._ACTIVATION_MAP[activation_function][1]

        # --- back prop variables ---
        self.layer_num = None
        self.input_cache = None
        self.unactivated_output_cache = None

        # --- metrics tracking variables---
        self.output_gradient_magnitude = []
        self.output_gradient_extremes = []
        self.activation_magnitude = []
        self.activation_extremes = []

    def build(self, input_shape: tuple):
        assert len(input_shape) == 4
        batch_size, in_height, in_width, in_channels = input_shape
        kernel_height, kernel_width, kernel_in_channels, out_channels = self.kernel_shape

        out_height = (in_height+self.padding[0] * 2 - kernel_height + 1) // self.stride[0]
        out_width = (in_width + self.padding[1] * 2 - kernel_width + 1) // self.stride[1]

        self.input_shape = input_shape
        self.output_shape = (batch_size, out_height, out_width, out_channels)

        if self.kernel is None:
            input_size = input_shape[3] * self.kernel_shape[0] * self.kernel_shape[1]
            output_size = self.kernel_shape[3] * self.kernel_shape[0] * self.kernel_shape[1]

            match self.kernel_initialization:
                case 'He' | 'Kaiming':
                    self.kernel = np.random.normal(0, scale=np.sqrt(2.0 / input_size), size=self.kernel_shape).astype(self.dtype)
                case 'Xavier' | 'Glorot':
                    self.kernel = np.random.normal(0, scale=np.sqrt(2.0 / (input_size + output_size)), size=self.kernel_shape).astype(self.dtype)
                case 'LeCun':
                    self.kernel = np.random.normal(0, scale=np.sqrt(1.0 / input_size), size=self.kernel_shape).astype(self.dtype)
                case 'swish':
                    self.kernel = np.random.normal(0, scale=1.1 / np.sqrt(input_size), size=self.kernel_shape).astype(self.dtype)
                case _:
                    raise Exception("initialization method not found or not implemented. Maybe check spelling?")

        if self.bias is None:
            match self.bias_initialization:
                case 'none':
                    self.bias = np.zeros(self.kernel_shape[3], dtype=self.dtype)

    def get_padding_obj(self, padding_type):
        match padding_type:
            case 'same':
                return [(self.kernel_shape[0] - 1) // 2, (self.kernel_shape[1] - 1) // 2]
            case 'valid':
                return [0, 0]
            case 'full':
                return [self.kernel_shape[0] - 1, self.kernel_shape[1] - 1]
            case 'none':
                return [0, 0]
            case _:
                raise Exception(f'unsupported padding type {padding_type}')

    @staticmethod
    def dilate(input_tensor: np.ndarray, dilation_rate: list):
        if dilation_rate == [1, 1]:
            return input_tensor

        # output_height = (input_tensor.shape[1] - 1) * dilation_rate[1] + 1
        # output_width = (input_tensor.shape[2] - 1) * dilation_rate[0] + 1
        output_height = input_tensor.shape[1] * dilation_rate[1]
        output_width = input_tensor.shape[2] * dilation_rate[0]
        dilated = np.zeros((input_tensor.shape[0], output_height, output_width, input_tensor.shape[3]))
        dilated[:, ::dilation_rate[1], ::dilation_rate[0], :] = input_tensor

        return dilated

    @staticmethod
    def cross_correlate2d(input_tensor: np.ndarray, kernel: np.ndarray, stride: list, padding: list, dtype=np.float32):
        @njit(parallel=True, fastmath=True, cache=True)
        def pad(x: np.ndarray, pad_h: int, pad_w: int):   # NHWC padding
            batch, in_h, in_w, channels = x.shape
            padded = np.zeros((batch, in_h + 2*pad_h, in_w + 2*pad_w, channels), dtype=x.dtype)

            for b in prange(batch):
                for c in prange(channels):
                    padded[b, pad_h:pad_h + in_h, pad_w:pad_w + in_w, c] = x[b, :, :, c]
            return padded

        """
        Args:
            input_tensor  : (batch_size, height, width, input_channels)
            kernel : (kernel_height, kernel_width, input_channels, output_channels)
            bias   : (output_channels,)
            stride : (vertical_stride, horizontal_stride)
            padding: (vertical_padding, horizontal_padding)
        Returns:
            output : (batch_size, output_height, output_width, output_channels)
        """

        # --- Dimensions ---
        batch_size, height, width, input_channels = input_tensor.shape
        kernel_height, kernel_width, _, output_channels = kernel.shape

        # --- Padding ---
        vertical_padding = padding[0]
        horizontal_padding = padding[1]

        padded_input = pad(input_tensor, vertical_padding, horizontal_padding)
        padded_height, padded_width = padded_input.shape[1], padded_input.shape[2]

        # --- Output Dimensions ---
        output_height = (padded_height - kernel_height) // stride[0] + 1
        output_width = (padded_width - kernel_width) // stride[1] + 1

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
            shape=(batch_size, output_height, output_width, kernel_height, kernel_width, input_channels),
            strides=strides,
            writeable=False
        )

        x_col = np.reshape(windows, (batch_size * output_height * output_width, kernel_height * kernel_width * input_channels), order='C')
        w_col = np.reshape(kernel, (kernel_height * kernel_width * input_channels, output_channels), order='F')
        output = np.dot(x_col.astype(dtype, copy=False), w_col.astype(dtype, copy=False))

        return output.reshape(batch_size, output_height, output_width, output_channels)

    def conv2d(self, input_tensor: np.ndarray, kernel: np.ndarray, stride: list, padding: list, dtype=np.float32):
        stride = [stride, stride] if isinstance(stride, int) else stride
        return self.cross_correlate2d(input_tensor, np.rot90(kernel, 2), stride, padding, dtype)

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0
        unactivated = self.cross_correlate2d(input_tensor, self.kernel, self.stride, self.padding, self.dtype) + self.bias
        activated = self.act_func(unactivated)
        self.input_cache = input_tensor
        self.unactivated_output_cache = unactivated

        return activated

    def backprop(self, output_gradient: np.ndarray, batch_size: int, lr: float, optimizer, clip_value: float):
        assert output_gradient.size != 0
        assert self.input_cache is not None
        assert self.unactivated_output_cache is not None

        # --- Partial Derivatives ---
        full_padding = self.get_padding_obj("full")
        di_padding = [full_padding[0]-self.padding[0], full_padding[1]-self.padding[1]]

        dilated_dz = self.dilate(output_gradient.astype(self.dtype) * self.d_act_func(self.unactivated_output_cache, dtype=self.dtype), self.stride)
        dw = self.cross_correlate2d(self.input_cache.transpose(3, 1, 2, 0), dilated_dz.transpose(1, 2, 0, 3), stride=[1, 1], padding=self.padding).transpose(1, 2, 0, 3)
        db = np.sum(dilated_dz, axis=(0, 1, 2), dtype=self.dtype)
        di = self.conv2d(dilated_dz, self.kernel.transpose(0, 1, 3, 2), stride=[1, 1], padding=di_padding)

        # --- Metrics Tracking ---
        self.activation_magnitude = np.mean(np.abs(self.unactivated_output_cache))
        self.activation_extremes = np.max(self.unactivated_output_cache) + np.abs(np.min(self.unactivated_output_cache)) / 2
        self.output_gradient_magnitude = np.mean(np.abs(output_gradient))
        self.output_gradient_extremes = np.max(output_gradient) + np.abs(np.min(output_gradient)) / 2

        #  --- Weights And Biases Update ---
        weight_change, bias_change = optimizer.adjust_gradient(self.layer_num, dw/batch_size, db/batch_size, lr, self.dtype)
        self.kernel -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        # --- Reset Caches ---
        self.input_cache = None
        self.unactivated_output_cache = None

        return di


class Dense:
    def __init__(self, size: int, activation_function='relu', weight_initialization='He', bias_initialization='none', dtype=np.float32):
        """ input_tensor  : (batchsize, input_size)
            weights : (input_size, output_size)
            bias   : (output_size)
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer")

        valid_activations = ['relu', 'sigmoid', 'tanh', 'swish', 'softmax']
        if activation_function not in valid_activations:
            raise ValueError(f"Invalid activation function. Must be one of {valid_activations}")

        self._ACTIVATION_MAP = {
            'relu': (ActivationFunction.relu, ActivationFunction.d_relu),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.d_sigmoid),
            'tanh': (ActivationFunction.tanh, ActivationFunction.d_tanh),
            'swish': (ActivationFunction.swish, ActivationFunction.d_swish),
            'softmax': (ActivationFunction.softmax, ActivationFunction.d_softmax),
        }

        self.type = 'dense'
        self.size = size
        self.input_shape = None
        self.output_shape = [-1, size]
        self.dtype = dtype

        self.weights = None
        self.bias = None

        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.activation_function = activation_function
        self.act_func = self._ACTIVATION_MAP[activation_function][0]
        self.d_act_func = self._ACTIVATION_MAP[activation_function][1]

        # --- back prop variables ---
        self.layer_num = None
        self.input_cache = None
        self.unactivated_output_cache = None

        # --- metrics tracking variables ---
        self.output_gradient_magnitude = []
        self.output_gradient_extremes = []
        self.activation_magnitude = []
        self.activation_extremes = []

    def build(self, input_shape: tuple):
        assert len(input_shape) == 2
        self.input_shape = input_shape

        if self.weights is None:
            output_size = self.size
            input_size = input_shape[1]

            match self.weight_initialization:
                case 'He' | 'Kaiming':
                    self.weights = np.random.normal(0, scale=np.sqrt(2.0 / input_size), size=(input_size, output_size)).astype(self.dtype)
                case 'Xavier' | 'Glorot':
                    self.weights = np.random.normal(0, scale=np.sqrt(2.0 / (input_size + output_size)), size=(input_size, output_size)).astype(self.dtype)
                case 'LeCun':
                    self.weights = np.random.normal(0, scale=np.sqrt(1.0 / input_size), size=(input_size, output_size)).astype(self.dtype)
                case 'swish':
                    self.weights = np.random.normal(0, scale=1.1 / np.sqrt(input_size), size=(input_size, output_size)).astype(self.dtype)
                case _:
                    raise Exception("initialization method not found or not implemented. Maybe check spelling?")

        if self.bias is None:
            match self.bias_initialization:
                case 'none':
                    self.bias = np.zeros((1, self.size), dtype=self.dtype)

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0

        unactivated = np.dot(input_tensor.astype(self.dtype), self.weights) + self.bias
        activated = self.act_func(unactivated)
        self.input_cache = input_tensor
        self.unactivated_output_cache = unactivated

        return activated

    def backprop(self, output_gradient: np.ndarray, batch_size: int, lr: float, optimizer, clip_value: float):
        assert output_gradient.size != 0
        assert self.input_cache is not None
        assert self.unactivated_output_cache is not None

        # --- Partial Derivatives ---
        dz = output_gradient.astype(self.dtype) * self.d_act_func(self.unactivated_output_cache, dtype=self.dtype)
        dw = np.dot(self.input_cache.T.astype(self.dtype), dz)
        db = np.sum(dz, axis=0, keepdims=True)
        di = np.dot(dz, self.weights.T)

        # --- Metrics Tracking ---
        self.activation_magnitude = np.mean(np.abs(self.unactivated_output_cache))
        self.activation_extremes = np.max(self.unactivated_output_cache) + np.abs(np.min(self.unactivated_output_cache)) / 2
        self.output_gradient_magnitude = np.mean(np.abs(output_gradient))
        self.output_gradient_extremes = np.max(output_gradient) + np.abs(np.min(output_gradient)) / 2

        #  --- Weights And Biases Update ---
        weight_change, bias_change = optimizer.adjust_gradient(self.layer_num, dw/batch_size, db/batch_size, lr, self.dtype)
        self.weights -= np.clip(weight_change, -clip_value, clip_value)
        self.bias -= np.clip(bias_change, -clip_value, clip_value)

        # --- Reset Caches ---
        self.input_cache = None
        self.unactivated_output_cache = None

        return di


class Pooling:
    def __init__(self, kernel_size: int, stride: list, padding: str, pool_mode="max"):
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")

        valid_pool_modes = ['max', 'average']
        if pool_mode not in valid_pool_modes:
            raise ValueError(f"Invalid pool_mode. Must be one of {valid_pool_modes}")

        self.type = 'pooling'
        self.stride = [stride, stride] if isinstance(stride, int) else stride
        self.layer_type = "pooling"
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        self.padding = self.get_padding_obj(padding)
        self.input_shape = None
        self.output_shape = None

        # --- for backprop ---
        self.input_data = None
        self.prev_layer = None
        self.argmax_indexes = None

    def get_padding_obj(self, padding_type):
        match padding_type:
            case 'same':
                return [(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2]
            case 'valid':
                return [0, 0]
            case 'full':
                return [self.kernel_size - 1, self.kernel_size - 1]
            case 'none':
                return [0, 0]
            case _:
                raise Exception(f'unsupported padding type {padding_type}')

    @staticmethod
    def pool(input_tensor: np.ndarray, kernel_size: int, stride: list, padding: list, pool_mode='max'):
        @njit(parallel=True, fastmath=True, cache=True)
        def pad(x: np.ndarray, pad_h: int, pad_w: int):   # NHWC padding
            batch, in_h, in_w, channels = x.shape
            padded = np.zeros((batch, in_h + 2 * pad_h, in_w + 2 * pad_w, channels), dtype=x.dtype)

            for b in prange(batch):
                for c in prange(channels):
                    padded[b, pad_h:pad_h + in_h, pad_w:pad_w + in_w, c] = x[b, :, :, c]
            return padded
        """
        Args:
            input_tensor  : (batch_size, height, width, input_channels)
            kernel : (kernel_size, kernel_size)
            bias   : (output_channels,)
            stride : (vertical_stride, horizontal_stride)
            padding: (vertical_padding, horizontal_padding)
        Returns:
            output : (batch_size, output_height, output_width, output_channels)
        """
        # --- Dimensions ---
        batch_size, height, width, input_channels = input_tensor.shape
        kernel_height = kernel_width = kernel_size

        # --- Padding ---
        vertical_padding = padding[0]
        horizontal_padding = padding[1]

        padded_input = pad(input_tensor, vertical_padding, horizontal_padding)
        padded_height, padded_width = padded_input.shape[1], padded_input.shape[2]

        # --- Output Dimensions ---
        output_height = (padded_height - kernel_height) // stride[0] + 1
        output_width = (padded_width - kernel_width) // stride[1] + 1

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
            shape=(batch_size, output_height, output_width, kernel_height, kernel_width, input_channels),
            strides=strides,
            writeable=False
        )

        if pool_mode == 'max':
            indexes = np.tile(np.arange(0, output_height*output_width*stride[0], stride[0]), batch_size*input_channels).reshape(batch_size, output_height, output_width, input_channels)
            indexes += np.argmax(windows.reshape(batch_size, output_height, output_width, kernel_height*kernel_width, input_channels), axis=3)
            return np.max(windows, axis=(3, 4))
        elif pool_mode == 'average':
            return np.average(windows, axis=(3, 4))
        else:
            raise Exception(f"unknown pool mode {pool_mode}")

    def build(self, input_shape):
        batch_size, in_height, in_width, in_channels = input_shape
        out_height = (in_height+self.padding[0] * 2 - self.kernel_size + 1) // self.stride[0]
        out_width = (in_width + self.padding[1] * 2 - self.kernel_size + 1) // self.stride[1]

        self.output_shape = (-1, out_height, out_width, in_channels)

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0
        return self.pool(input_tensor, self.kernel_size, self.stride, self.padding, self.pool_mode)

    def backprop(self, output_gradient: np.ndarray):
        assert output_gradient.size != 0
        if self.pool_mode == 'max':
            return output_gradient
        elif self.pool_mode == 'average':
            return output_gradient
        else:
            raise Exception(f"unknown pool mode {self.pool_mode}")


class Reshape:
    def __init__(self, output_shape: tuple):
        self.type = 'reshape'
        self.layer_num = None
        self.input_shape = None
        self.output_shape = output_shape

    def build(self, input_shape: tuple):
        self.input_shape = input_shape

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0
        assert input_tensor.shape[1:] == self.input_shape[1:]
        return input_tensor.reshape(self.output_shape)

    def backprop(self, output_gradient: np.ndarray):
        assert output_gradient.size != 0
        return output_gradient.reshape(self.input_shape)


class Flatten:
    def __init__(self):
        self.type = 'flatten'
        self.layer_num = None
        self.input_shape = None
        self.output_shape = None

    def build(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], int(np.prod(input_shape[1:])))

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0
        return input_tensor.reshape(self.output_shape)

    def backprop(self, output_gradient: np.ndarray):
        assert output_gradient.size != 0
        return output_gradient.reshape(self.input_shape)


class Dropout:
    def __init__(self, dropout_rate: float):
        self.type = 'dropout'
        self.layer_num = None
        self.dropout_rate = dropout_rate

    def forprop(self, input_tensor: np.ndarray):
        assert input_tensor.size != 0

        binary_tensor = np.random.rand(*input_tensor.shape[1:]) <= (1 - self.dropout_rate)
        return input_tensor*binary_tensor/(1 - self.dropout_rate)
