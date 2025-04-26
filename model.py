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
        self.input_shape = None

    @staticmethod
    def get_optimizer(optimizer: str):
        match optimizer:
            case 'Adam':
                return Adam()
            case 'none':
                return NoOptimizer()
            case _:
                raise Exception(f"no optimizer called {optimizer}")

    def build(self, input_shape, optimizer='Adam', loss_func='cce'):
        self.optimizer = optimizer
        self.optimizer_obj = self.get_optimizer(optimizer)
        self.loss_func = loss_func
        self.input_shape = input_shape

        layer_num = 0
        for layer in self.layers:
            match layer.type:
                case "dense":
                    layer.build(input_shape)
                    if optimizer == "Adam":
                        self.optimizer_obj.fme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                        self.optimizer_obj.sme[layer_num] = [np.zeros(layer.weights.shape), np.zeros(layer.bias.shape)]
                case "convolution":
                    layer.build(input_shape)
                    if optimizer == "Adam":
                        self.optimizer_obj.fme[layer_num] = [np.zeros(layer.kernel.shape), np.zeros(layer.bias.shape)]
                        self.optimizer_obj.sme[layer_num] = [np.zeros(layer.kernel.shape), np.zeros(layer.bias.shape)]
                case "pooling":
                    layer.build(input_shape)
                case "reshape":
                    layer.build(input_shape)
                case "flatten":
                    layer.build(input_shape)
                case "dropout":
                    pass
                case _:
                    raise Exception(f"unknown layer type {layer.type}")

            layer.layer_num = layer_num
            input_shape = layer.output_shape
            layer_num += 1

    def save(self, directory: str, file_name: str):
        layer_data = {'layer_num': len(self.layers),
                      'optimizer': self.optimizer,
                      'loss_func': self.loss_func,
                      'input_shape': self.input_shape
                      }

        for i in range(len(self.layers)):
            match self.layers[i].type:
                case 'dense':
                    layer_data[f'layer{i}'] = 'dense'
                    layer_data[f'layer{i}_dtype'] = f"np.{self.layers[i].dtype.__name__}"
                    layer_data[f'layer{i}_weights_init'] = self.layers[i].weight_initialization
                    layer_data[f'layer{i}_bias_init'] = self.layers[i].bias_initialization
                    layer_data[f'layer{i}_weights'] = self.layers[i].weights
                    layer_data[f'layer{i}_bias'] = self.layers[i].bias
                    layer_data[f'layer{i}_activation_func'] = self.layers[i].activation_function
                    layer_data[f'layer{i}_input_shape'] = self.layers[i].input_shape
                case 'convolution':
                    layer_data[f'layer{i}'] = 'convolution'
                    layer_data[f'layer{i}_dtype'] = f"np.{self.layers[i].dtype.__name__}"
                    layer_data[f'layer{i}_kernel_init'] = self.layers[i].kernel_initialization
                    layer_data[f'layer{i}_bias_init'] = self.layers[i].bias_initialization
                    layer_data[f'layer{i}_kernel'] = self.layers[i].kernel
                    layer_data[f'layer{i}_bias'] = self.layers[i].bias
                    layer_data[f'layer{i}_padding'] = self.layers[i].padding
                    layer_data[f'layer{i}_stride'] = self.layers[i].stride
                    layer_data[f'layer{i}_activation_func'] = self.layers[i].activation_function
                    layer_data[f'layer{i}_input_shape'] = self.layers[i].input_shape
                case 'pooling':
                    layer_data[f'layer{i}'] = 'pooling'
                    layer_data[f'layer{i}_kernel_size'] = self.layers[i].kernel_size
                    layer_data[f'layer{i}_stride'] = self.layers[i].stride
                    layer_data[f'layer{i}_padding'] = self.layers[i].padding
                    layer_data[f'layer{i}_pool_mode'] = self.layers[i].pool_mode
                case 'reshape':
                    layer_data[f'layer{i}'] = 'reshape'
                    layer_data[f'layer{i}_output_shape'] = self.layers[i].output_shape
                case 'flatten':
                    layer_data[f'layer{i}'] = 'flatten'
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
                    new_layer.dtype = eval(str(data[f'layer{i}_dtype']))
                    new_layer.weight_initialization = str(data[f'layer{i}_weights_init'])
                    new_layer.bias_initialization = str(data[f'layer{i}_bias_init'])
                    new_layer.input_shape = tuple(data[f'layer{i}_input_shape'].tolist())
                    new_layer.weights = data[f'layer{i}_weights']
                    new_layer.bias = data[f'layer{i}_bias']
                    new_layer.activation_function = str(data[f'layer{i}_activation_func'])
                case 'convolution':
                    new_layer = layers.Convolution(data[f'layer{i}_kernel'].shape)
                    new_layer.dtype = eval(str(data[f'layer{i}_dtype']))
                    new_layer.kernel_initialization = str(data[f'layer{i}_kernel_init'])
                    new_layer.bias_initialization = str(data[f'layer{i}_bias_init'])
                    new_layer.input_shape = tuple(data[f'layer{i}_input_shape'])
                    new_layer.kernel = data[f'layer{i}_kernel']
                    new_layer.bias = data[f'layer{i}_bias']
                    new_layer.padding = data[f'layer{i}_padding'].tolist()
                    new_layer.stride = data[f'layer{i}_stride'].tolist()
                    new_layer.activation_function = str(data[f'layer{i}_activation_func'])
                case 'pooling':
                    kernel_size = data[f'layer{i}_kernel_size'].tolist()
                    stride = data[f'layer{i}_stride'].tolist()
                    padding = data[f'layer{i}_padding'].tolist()
                    pool_mode = str(data[f'layer{i}_pool_mode'])
                    new_layer = layers.Pooling(kernel_size, stride, padding, pool_mode)
                case 'reshape':
                    output_shape = tuple(data[f'layer{i}_output_shape'].tolist())
                    new_layer = layers.Reshape(output_shape)
                case 'flatten':
                    new_layer = layers.Flatten()
                case 'dropout':
                    dropout_rate = float(data[f'layer{i}_dropout_rate'])
                    new_layer = layers.Dropout(dropout_rate)
                case _:
                    raise Exception(f'unsupported layer type at layer {i}')

            self.layers.append(new_layer)

        self.build(tuple(data['input_shape'].tolist()), str(data['optimizer']), str(data['loss_func']))

    def forprop(self, input_tensor: np.ndarray, mode='training'):
        batch_size = input_tensor.shape[0]
        for layer in self.layers:
            assert input_tensor.shape[0] == batch_size

            match layer.type:
                case 'dense':
                    input_tensor = layer.forprop(input_tensor)
                case 'convolution':
                    input_tensor = layer.forprop(input_tensor)
                case 'pooling':
                    input_tensor = layer.forprop(input_tensor)
                case 'dropout':
                    if mode == 'testing':
                        continue
                    input_tensor = layer.forprop(input_tensor)
                case 'reshape':
                    input_tensor = layer.forprop(input_tensor)
                case 'flatten':
                    input_tensor = layer.forprop(input_tensor)
                case _:
                    raise Exception(f'unknown layer type {layer.type}')

        return input_tensor

    def backprop(self, output_gradient: np.ndarray, batch_size: int, learning_rate: float, optimizer, clip_value: float):
        for layer in reversed(self.layers):
            # back propagates, updates weight/bias changes, and outputs input gradient
            match layer.type:
                case 'dense':
                    output_gradient = layer.backprop(output_gradient, batch_size, learning_rate, optimizer, clip_value)
                case 'convolution':
                    output_gradient = layer.backprop(output_gradient, batch_size, learning_rate, optimizer, clip_value)
                case 'dropout':
                    pass
                case 'reshape':
                    output_gradient = layer.backprop(output_gradient)
                case 'flatten':
                    output_gradient = layer.backprop(output_gradient)
                case _:
                    raise Exception(f'unknown layer type {layer.type}')

    def train(self, data: Data, epochs: int, batch_size: int, learning_rate: float, clip_value: float, tracker: MetricTracker, readout_freq=10, readout_sample_size=100, dtype=np.float32):
        assert readout_sample_size < batch_size

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
        validation_loss = 0
        validation_accuracy = 0

        batches_per_epoch = data.training_data.shape[0] // batch_size
        for epoch in range(epochs):
            data.shuffle('training')
            for batch in range(batches_per_epoch):
                # --- validation testing ---
                if batch % readout_freq == 0:
                    validation_indexes = np.random.choice(np.arange(data.validation_labels.shape[0]), size=readout_sample_size, replace=False)
                    validation_labels = data.validation_labels[validation_indexes]
                    validation_data = data.validation_data[validation_indexes]
                    validation_predictions = self.forprop(validation_data, mode='testing')
                    validation_loss = -np.sum(validation_labels * np.log(np.clip(validation_predictions, 1e-7, 1 - 1e-7))) / readout_sample_size
                    validation_accuracy = np.sum(np.argmax(validation_predictions, axis=1) == np.argmax(validation_labels, axis=1)) / readout_sample_size

                # --- predictions and backprop ---
                training_labels = data.training_labels[batch * batch_size:(batch + 1) * batch_size]
                training_data = data.training_data[batch * batch_size:(batch + 1) * batch_size]
                training_predictions = self.forprop(training_data)
                self.backprop(training_predictions - training_labels, batch_size, learning_rate, self.optimizer_obj, clip_value)

                # --- normal testing ---
                training_indexes = np.random.choice(np.arange(training_data.shape[0]), size=readout_sample_size, replace=False)
                training_predictions = self.forprop(training_data[training_indexes], mode='testing')  # rerun predictions without dropout
                training_labels = training_labels[training_indexes]
                loss = -np.sum(training_labels * np.log(np.clip(training_predictions, 1e-7, 1 - 1e-7))) / readout_sample_size
                training_accuracy = np.sum(np.argmax(training_predictions, axis=1) == np.argmax(training_labels, axis=1)) / readout_sample_size
                tracker.performance_metrics_update(loss, training_accuracy)

                # --- tracking ---
                activation_magnitude = []
                activation_extremes = []
                output_gradient_magnitude = []
                output_gradient_extremes = []
                for layer in self.layers:
                    if layer.type not in ['reshape', 'dropout', 'flatten']:
                        activation_magnitude.append(layer.activation_magnitude)
                        activation_extremes.append(layer.activation_extremes)
                        output_gradient_magnitude.append(layer.output_gradient_magnitude)
                        output_gradient_extremes.append(layer.output_gradient_extremes)
                tracker.training_metrics_update(np.mean(activation_magnitude), np.mean(activation_extremes), np.mean(output_gradient_magnitude), np.mean(output_gradient_extremes))

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
