import numpy as np
import layers


class SequentialModel:
    def __init__(self, input_size=None):
        self.layers = []
        self.input_size = input_size
        self.optimizer = None
        self.loss_func = None
        self.metrics = None

    def build(self, optimizer=None, loss_func='mse', metrics=['accuracy']):
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = metrics
        input_size = self.input_size
        for layer in self.layers:
            layer.build(input_size)
            input_size = layer.output_size

    def save(self, directory, file_name):
        layer_data = {'layer_num': len(self.layers),
                      'input_size': self.input_size,
                      'optimizer': self.optimizer,
                      'loss_func': self.loss_func,
                      'metrics': self.metrics
                      }

        for i in range(len(self.layers)):
            layer_data[f'layer{i}'] = 'dense'
            layer_data[f'layer{i}_weights'] = self.layers[i].weights
            layer_data[f'layer{i}_bias'] = self.layers[i].bias
            layer_data[f'layer{i}_activation_func'] = self.layers[i].activation_function

        np.savez_compressed(f'{directory}/{file_name}.npz', **layer_data)

    def load(self, file_location):
        data = np.load(file_location, allow_pickle=True)
        optimizer = data['optimizer']
        loss_func = data['loss_func']
        metrics = data['metrics']

        self.layers = []
        for i in range(data['layer_num']):
            match data[f'layer{i}']:
                case 'dense':
                    size = data[f'layer{i}_weights'].shape[0]
                    new_layer = layers.Dense(size)
                    new_layer.weights = data[f'layer{i}_weights']
                    new_layer.bias = data[f'layer{i}_bias']
                    new_layer.activation_function = data[f'layer{i}_activation_func']
                case _:
                    raise Exception(f'unsupported layer type at layer {i}')
            self.layers.append(new_layer)

        self.build(optimizer, loss_func, metrics)
        self.input_size = data['input_size']

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backprop(self):
        return
