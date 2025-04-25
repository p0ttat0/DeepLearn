import numpy as np


class ActivationFunction:
    @staticmethod
    def relu(x: np.ndarray, dtype=np.float32):
        return np.maximum(0, x.astype(dtype))

    @staticmethod
    def sigmoid(x: np.ndarray, dtype=np.float32):
        return 1 / (1 + np.exp(-x, dtype=dtype))

    @staticmethod
    def tanh(x: np.ndarray, dtype=np.float32):
        return np.tanh(x, dtype=dtype)

    @staticmethod
    def swish(x: np.ndarray, dtype=np.float32):
        return x.astype(dtype) / (1 + np.exp(-x, dtype=dtype))

    @staticmethod
    def softmax(x: np.ndarray, dtype=np.float32):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True), dtype=dtype)
        return e_x / np.sum(e_x, axis=1, keepdims=True, dtype=dtype)

    @staticmethod
    def d_relu(x: np.ndarray, dtype=np.uint8):
        return (x > 0).astype(dtype)

    @staticmethod
    def d_sigmoid(x: np.ndarray, dtype=np.float32):
        activated = 1 / (1 + np.exp(-x, dtype=dtype))
        return activated * (1 - activated)

    @staticmethod
    def d_tanh(x: np.ndarray, dtype=np.float32):
        activated = np.tanh(x, dtype=dtype)
        return 1 - np.square(activated, dtype=dtype)

    @staticmethod
    def d_swish(x: np.ndarray, dtype=np.float32):
        sig = 1 / (1 + np.exp(-x, dtype=dtype))
        return sig * (1 + x.astype(dtype) * (1 - sig))

    @staticmethod
    # assumes categorical cross entropy
    def d_softmax(x: np.ndarray, dtype=np.float32):
        return np.ones_like(x, dtype=dtype)
