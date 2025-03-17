from model import SequentialModel
from layers import Dense, Flatten
from data import Dataset


training_data, training_labels, validation_data, validation_labels = Dataset.mnist()

my_model = SequentialModel()
my_model.layers = [
    Flatten(input_shape=[-1, 28, 28]),
    Dense(120, activation_function="swish", weight_initialization='He'),
    Dense(100, activation_function="swish", weight_initialization='He'),
    Dense(40, activation_function="swish", weight_initialization='He'),
    Dense(10, activation_function='swish', weight_initialization='He'),
]

my_model.build()
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))
# my_model.save('saved models', 'model')
my_model.load('saved models/model.npz')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))

my_model.train(training_data, training_labels, 50, 1200, 0.001)

my_model.save('saved models', 'model')
