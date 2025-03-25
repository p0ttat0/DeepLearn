from model import SequentialModel
from layers import Dense, Flatten
from data import Dataset


training_data, training_labels, validation_data, validation_labels = Dataset.mnist()

my_model = SequentialModel()
my_model.layers = [
    Flatten(input_shape=[-1, 28, 28]),
    Dense(180, "swish", 'LeCun'),
    Dense(120, "swish", 'LeCun'),
    Dense(60, "swish", 'LeCun'),
    Dense(30, "swish", 'LeCun'),
    Dense(10, 'softmax', 'Xavier'),
]

my_model.build(optimizer='none')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))
# my_model.save('saved models', 'model1')
# my_model.load('saved models/model1.npz')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))

my_model.train(training_data, training_labels, 2, 600, 0.01, clip_value=0.5)

my_model.save('saved models', 'model1')
