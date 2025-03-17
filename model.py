import numpy as np
import layers
from progress_bar import ProgressBar

class SequentialModel:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss_func = None

    def build(self, optimizer=None, loss_func='mse'):
        self.optimizer = optimizer
        self.loss_func = loss_func
        input_shape = self.layers[0].input_shape
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.output_shape

    def save(self, directory, file_name):
        layer_data = {'layer_num': len(self.layers),
                      'optimizer': self.optimizer,
                      'loss_func': self.loss_func,
                      }

        for i in range(len(self.layers)):
            match self.layers[i].type:
                case 'dense':
                    layer_data[f'layer{i}'] = 'dense'
                    layer_data[f'layer{i}_weights_init'] = self.layers[i].weight_initialization
                    layer_data[f'layer{i}_bias_init'] = self.layers[i].bias_initialization
                    layer_data[f'layer{i}_weights'] = self.layers[i].weights
                    layer_data[f'layer{i}_bias'] = self.layers[i].bias
                    layer_data[f'layer{i}_activation_func'] = self.layers[i].activation_function
                    layer_data[f'layer{i}_input_shape'] = self.layers[i].input_shape
                case 'flatten':
                    layer_data[f'layer{i}'] = 'flatten'
                    layer_data[f'layer{i}_input_shape'] = self.layers[i].input_shape
                case _:
                    raise Exception(f'unsupported layer of type {self.layers[i].type}')

        np.savez_compressed(f'{directory}/{file_name}.npz', **layer_data)

    def load(self, file_location):
        data = np.load(file_location, allow_pickle=True)
        self.layers = []

        for i in range(data['layer_num']):
            match data[f'layer{i}']:
                case 'dense':
                    size = data[f'layer{i}_weights'].shape[0]

                    new_layer = layers.Dense(size)
                    new_layer.weight_initialization = str(data[f'layer{i}_weights_init'])
                    new_layer.bias_initialization = str(data[f'layer{i}_bias_init'])
                    new_layer.input_shape = data[f'layer{i}_input_shape'].tolist()
                    new_layer.weights = data[f'layer{i}_weights']
                    new_layer.bias = data[f'layer{i}_bias']
                    new_layer.activation_function = str(data[f'layer{i}_activation_func'])
                case 'flatten':
                    new_layer = layers.Flatten(data[f'layer{i}_input_shape'].tolist())
                case _:
                    raise Exception(f'unsupported layer type at layer {i}')

            self.layers.append(new_layer)

        self.build(data['optimizer'], data['loss_func'])

    def forprop(self, x):
        for layer in self.layers:
            x = layer.forprop(x)
        return x

    def backprop(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backprop(output_gradient)

    def train(self, training_data, training_labels, epochs, batch_size, learning_rate):
        progress = ProgressBar()
        progress.start()
        training_examples = training_data.shape[0]
        batches_per_epoch = training_examples // batch_size

        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                labels = training_labels[:, batch * batch_size:(batch + 1) * batch_size]
                data = training_data[batch * batch_size:(batch + 1) * batch_size]

                predictions = self.forprop(data)
                loss = -np.sum(labels * np.log(np.clip(predictions, 1e-7, 1 - 1e-7)))/batch_size
                training_accuracy = np.sum(np.argmax(predictions, axis=0) == np.argmax(labels, axis=0))/batch_size

                output_gradient = predictions-labels
                self.backprop(output_gradient)

                progress.update(epochs, batch+epoch*batches_per_epoch, batches_per_epoch, training_accuracy, loss)

                '''y = np.random.randint(0, batch_size-1)
                print(np.argmax(labels[:, y]))
                from matplotlib import pyplot as plt
                plt.imshow(training_data[data_start:data_end][y], interpolation='nearest')
                plt.show()'''

            for layer in self.layers:
                layer.apply_changes(batch_size, learning_rate)
