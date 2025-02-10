import numpy as np


class Sequential:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = None
        self. metrics = None

    def compile(self, input_size=1, optimizer=None, loss='mse', metrics=['accuracy']):
        self.optimizer = optimizer
        self.loss = loss
        self. metrics = metrics
        for layer in self.layers:
            layer.compile(input_size)
            input_size = layer.size

    def save(self, directory, file_name):
        return

    def load(self, file_location):
        return

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backprop(self):
        return
