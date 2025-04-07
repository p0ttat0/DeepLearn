import numpy as np
import layers
from progress_bar import ProgressBar
from optimizers import Adam, NoOptimizer
from pynput import keyboard
from tracking import MetricTracker
from data import Data


class SequentialModel:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.optimizer_obj = None
        self.loss_func = None

    @staticmethod
    def get_optimizer(optimizer: str):
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
        input_shape = self.layers[0].output_shape
        layer_num = 0
        for layer in self.layers:
            if layer.type not in ['reshape', 'dropout']:
                layer.build(input_shape)
                layer.layer_num = layer_num

                if optimizer == "Adam":
                    self.optimizer_obj.fme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                    self.optimizer_obj.sme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                input_shape = layer.output_shape
            layer_num += 1

    def save(self, directory: str, file_name: str):
        layer_data = {'layer_num': len(self.layers),
                      'optimizer': self.optimizer,
                      'optimizer_step': self.optimizer_obj.step,
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
                case 'reshape':
                    layer_data[f'layer{i}'] = 'reshape'
                    layer_data[f'layer{i}_input_shape'] = self.layers[i].input_shape
                    layer_data[f'layer{i}_output_shape'] = self.layers[i].output_shape
                case 'dropout':
                    layer_data[f'layer{i}'] = 'dropout'
                    layer_data[f'layer{i}_dropout_rate'] = self.layers[i].dropout_rate
                case _:
                    raise Exception(f'unsupported layer of type {self.layers[i].type}')

        np.savez_compressed(f'{directory}/{file_name}.npz', **layer_data)

    def load(self, file_location: str):
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
                case 'reshape':
                    new_layer = layers.Reshape(data[f'layer{i}_input_shape'].tolist(), data[f'layer{i}_output_shape'].tolist())
                case 'dropout':
                    new_layer = layers.Dropout(float(data[f'layer{i}_dropout_rate']))
                case _:
                    raise Exception(f'unsupported layer type at layer {i}')

            self.layers.append(new_layer)

        self.build(str(data['optimizer']), str(data['loss_func']))
        self.optimizer_obj.step = int(data['optimizer_step'])

    def forprop(self, input_tensor: np.ndarray, mode='training'):
        for layer in self.layers:
            if mode == 'testing' and layer.type in ['dropout']:
                continue
            input_tensor = layer.forprop(input_tensor)
        return input_tensor

    def backprop(self, output_gradient: np.ndarray):
        for layer in reversed(self.layers):
            # back propagates to accumulate weight/bias changes and outputs input gradient
            output_gradient = layer.backprop(output_gradient)

    def train(self, data: Data, epochs: int, batch_size: int, learning_rate: float, clip_value: float, tracker: MetricTracker, readout_freq=10, readout_sample_size=100):
        def on_press(key):
            try:
                if key == keyboard.Key.f7:
                    nonlocal training
                    training = False
                    return False
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        progress_bar = ProgressBar()
        progress_bar.start()
        tracker.reset()
        training = True

        batches_per_epoch = data.training_data.shape[0] // batch_size
        for epoch in range(epochs):
            data.shuffle('training')
            for batch in range(batches_per_epoch):
                # validation
                if batch % readout_freq == 0:
                    indexes = np.random.choice(np.arange(data.validation_labels.shape[0]), size=readout_sample_size, replace=False)
                    validation_labels = data.validation_labels[indexes]
                    validation_data = data.validation_data[indexes]
                    validation_predictions = self.forprop(validation_data, mode='testing')
                    validation_loss = -np.sum(validation_labels * np.log(np.clip(validation_predictions, 1e-7, 1 - 1e-7))) / readout_sample_size
                    validation_accuracy = np.sum(np.argmax(validation_predictions, axis=1) == np.argmax(validation_labels, axis=1)) / readout_sample_size

                # predictions and backprop
                training_labels = data.training_labels[batch * batch_size:(batch + 1) * batch_size]
                training_data = data.training_data[batch * batch_size:(batch + 1) * batch_size]
                training_predictions = self.forprop(training_data)
                self.backprop(training_predictions - training_labels)

                # training metrics
                indexes = np.random.choice(np.arange(training_data.shape[0]), size=readout_sample_size, replace=False)
                training_predictions = self.forprop(training_data[indexes], mode='testing')  # rerun predictions without dropout
                training_labels = training_labels[indexes]
                loss = -np.sum(training_labels * np.log(np.clip(training_predictions, 1e-7, 1 - 1e-7))) / readout_sample_size
                training_accuracy = np.sum(np.argmax(training_predictions, axis=1) == np.argmax(training_labels, axis=1)) / readout_sample_size
                tracker.performance_metrics_update(loss, training_accuracy)

                # applies changes
                for layer in self.layers:
                    if layer.type not in ['reshape', 'dropout']:
                        layer.apply_changes(batch_size, learning_rate, self.optimizer_obj, clip_value)
                        tracker.training_metrics_update(layer.activation_magnitude, layer.activation_extremes, layer.output_gradient_magnitude, layer.output_gradient_extremes)

                progress_bar.update(epochs, batch + epoch * batches_per_epoch, batches_per_epoch, training_accuracy, loss, validation_loss, validation_accuracy)
                self.optimizer_obj.step += 1

                if not training:
                    print("\nF7 pressed. Exiting...")
                    break
            else:
                continue
            break

        progress_bar.end()
        tracker.show()

    def test(self, data: np.ndarray, labels: np.ndarray, iterations):
        import matplotlib.pyplot as plt

        for i in range(iterations):
            x = np.random.randint(0, data.shape[0])
            plt.imshow(data[x], cmap='viridis')
            plt.show()
            prediction = self.forprop(data[x], mode='testing')
            print(prediction)
            print("prediction: "+str(np.argmax(prediction, axis=1)[0])+"     label:"+str(np.argmax(labels[x], axis=0)))
