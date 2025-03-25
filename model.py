import numpy as np
import layers
from progress_bar import ProgressBar
from optimizers import Adam, NoOptimizer
from matplotlib import pyplot as plt


class SequentialModel:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.optimizer_obj = None
        self.loss_func = None

    @staticmethod
    def get_optimizer(optimizer):
        match optimizer:
            case 'Adam':
                return Adam()
            case 'none':
                return NoOptimizer()
            case _:
                raise Exception(f"no optimizer called {optimizer}")

    def build(self, optimizer='Adam', loss_func='cce'):
        self.optimizer = optimizer
        self.optimizer_obj = self.get_optimizer(optimizer)
        self.loss_func = loss_func
        input_shape = self.layers[0].input_shape
        layer_num = 0
        for layer in self.layers:
            layer.build(input_shape)
            layer.layer_num = layer_num
            input_shape = layer.output_shape

            if layer.type not in ['flatten'] and optimizer not in ['none']:
                self.optimizer_obj.fme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                self.optimizer_obj.sme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                self.optimizer_obj.prev_fme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                self.optimizer_obj.prev_sme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]

            layer_num += 1

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

        self.build(str(data['optimizer']), str(data['loss_func']))

    def forprop(self, x):
        for layer in self.layers:
            x = layer.forprop(x)
        return x

    def backprop(self, output_gradient, clip_value=1):
        for layer in reversed(self.layers):
            # back propagates to accumulate weight/bias changes and outputs input gradient
            output_gradient = layer.backprop(output_gradient, clip_value)

    def train(self, training_data, training_labels, epochs, batch_size, learning_rate, clip_value=0.5):
        progress = ProgressBar()
        progress.start()
        training_examples = training_data.shape[0]
        batches_per_epoch = training_examples // batch_size
        losses = []

        def shuffle_data(d, l):
            assert d.shape[0] == l.T.shape[0]
            p = np.random.permutation(d.shape[0])
            return d[p], l.T[p].T

        for epoch in range(epochs):
            training_data, training_labels = shuffle_data(training_data, training_labels)
            for batch in range(batches_per_epoch):
                labels = training_labels[:, batch * batch_size:(batch + 1) * batch_size]
                data = training_data[batch * batch_size:(batch + 1) * batch_size]

                predictions = self.forprop(data)
                loss = -np.sum(labels * np.log(np.clip(predictions, 1e-7, 1 - 1e-7)))/batch_size
                training_accuracy = np.sum(np.argmax(predictions, axis=0) == np.argmax(labels, axis=0))/batch_size

                output_gradient = predictions-labels
                self.backprop(output_gradient, clip_value)

                losses.append(loss)
                progress.update(epochs, batch+epoch*batches_per_epoch, batches_per_epoch, training_accuracy, loss)

                for layer in self.layers:
                    layer.apply_changes(batch_size, learning_rate, self.optimizer_obj)

        progress.end()

        x, y = np.arange(len(losses)), losses
        plt.plot(x, y, 'o')
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.show()
