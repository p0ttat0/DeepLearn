import os
import gzip
import numpy as np


class Data:
    def __init__(self, training_data, training_labels, validation_data, validation_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def shuffle(self, target):
        if target == 'training':
            assert self.training_data.shape[0] == self.training_labels.shape[0]
            p = np.random.permutation(self.training_data.shape[0])
            self.training_data = self.training_data[p]
            self.training_labels = self.training_labels[p]
        if target == 'validation':
            assert self.validation_data.shape[0] == self.validation_labels.shape[0]
            p = np.random.permutation(self.validation_data.shape[0])
            self.validation_data = self.validation_data[p]
            self.validation_labels = self.validation_labels[p]


class Dataset:
    @staticmethod
    def mnist():  # loads mnist and returns training data, training labels, testing data, testing labels
        directory = os.path.dirname(__file__)
        training_data_path = os.path.join(directory, r"mnist\train-images-idx3-ubyte.gz")
        training_labels_path = os.path.join(directory, r"mnist\train-labels-idx1-ubyte.gz")
        test_data_path = os.path.join(directory, r"mnist\t10k-images-idx3-ubyte.gz")
        test_label_path = os.path.join(directory, r"mnist\t10k-labels-idx1-ubyte.gz")

        f = gzip.open(training_data_path, 'r')
        f.read(16)  # reads buffer
        img_data = np.frombuffer(f.read(60000 * 784), dtype=np.uint8).reshape((60000, 784)) / 255 * 2 - 1
        f = gzip.open(training_labels_path, 'r')
        f.read(8)  # reads buffer
        img_labels = np.frombuffer(f.read(60000), dtype=np.uint8).astype(int)[:, np.newaxis]
        img_labels = np.eye(10)[img_labels].squeeze()  # one hot encoding

        f = gzip.open(test_data_path, 'r')
        f.read(16)  # reads buffer
        testing_data = np.frombuffer(f.read(10000 * 784), dtype=np.uint8).reshape((10000, 784)) / 255 * 2 - 1
        f = gzip.open(test_label_path, 'r')
        f.read(8)  # reads buffer
        testing_labels = np.frombuffer(f.read(10000), dtype=np.uint8).astype(int)[:, np.newaxis]  # 2d array
        testing_labels = np.eye(10)[testing_labels].squeeze()

        return Data(img_data.reshape((60000, 28, 28)), img_labels, testing_data.reshape((10000, 28, 28)),
                    testing_labels)

    @staticmethod
    def load_generic(training_data_path, training_labels_path,validation_data_path, validation_labels_path):
        pass
