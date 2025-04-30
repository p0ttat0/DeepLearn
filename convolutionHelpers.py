import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange


def get_windows(input_tensor: np.ndarray, kernel_shape: tuple, stride: list):
    """Extracts sliding windows from input tensor for convolution.
    Args:
        input_tensor: Shape (batch, height, width, channels_in).
        kernel_shape: Tuple (kernel_h, kernel_w, channels_in, channels_out).
        stride: List [stride_h, stride_w].
    Returns:
        np.ndarray: Windows of shape (batch, out_h, out_w, kernel_h, kernel_w, channels_in).
    """
    # --- Dimensions ---
    batch_size, height, width, input_channels = input_tensor.shape
    kernel_height, kernel_width, _, output_channels = kernel_shape

    # --- Output Dimensions ---
    output_height = (height - kernel_height + 1) // stride[0]
    output_width = (width - kernel_width + 1) // stride[1]

    batch_stride, height_stride, width_stride, channel_stride = input_tensor.strides
    strides = (
        batch_stride,
        height_stride * stride[0],
        width_stride * stride[1],
        height_stride,
        width_stride,
        channel_stride
    )

    windows = as_strided(
        input_tensor,
        shape=(batch_size, output_height, output_width, kernel_height, kernel_width, input_channels),
        strides=strides,
        writeable=False
    )

    return windows


@njit(parallel=True, fastmath=True, cache=True)
def pad(x: np.ndarray, padding: list):  # NHWC padding
    pad_h, pad_w = padding
    batch, in_h, in_w, channels = x.shape
    padded = np.zeros((batch, in_h + 2 * pad_h, in_w + 2 * pad_w, channels), dtype=x.dtype)

    for batch_idx in prange(batch):
        for channel_idx in prange(channels):
            padded[batch_idx, pad_h:pad_h + in_h, pad_w:pad_w + in_w, channel_idx] = x[batch_idx, :, :, channel_idx]

    return padded
