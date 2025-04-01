from model import SequentialModel
from layers import Dense, Reshape
from data import Dataset
from tracking import MetricTracker
import numpy as np

data = Dataset.mnist()

my_model = SequentialModel()

my_model.layers = [
    Reshape(input_shape=[-1, 28, 28], output_shape=[-1, 784]),
    Dense(128, "relu", 'He'),
    Dense(64, "relu", 'He'),
    Dense(10, 'softmax', 'Xavier'),
]

my_model.build(optimizer='Adam')
my_model.forprop(np.random.rand(12, 28, 28))
my_model.save('saved models', 'test')
my_model.load('saved models/test')
my_model.forprop(np.random.rand(12, 28, 28))

monitor = MetricTracker(my_model, ['training accuracy', 'training losses', 'gradient magnitude', 'gradient extremes', 'activation magnitude'])

my_model.train(
    data,
    epochs=10,
    batch_size=128,
    learning_rate=0.001,
    clip_value=10,
    tracker=monitor
)
my_model.test(data.validation_data, data.validation_labels, 10)
# my_model.save('saved models', 'model1')
