from model import SequentialModel
from layers import Dense, Reshape, Dropout
from data import Dataset
from tracking import MetricTracker
import numpy as np

data = Dataset.mnist()

my_model = SequentialModel()

my_model.layers = [
    Reshape(input_shape=[-1, 28, 28], output_shape=[-1, 784]),
    Dropout(0.2),
    Dense(256, "swish", 'He'),
    Dropout(0.1),
    Dense(128, "swish", 'He'),
    Dropout(0.1),
    Dense(64, "swish", 'He'),
    Dense(10, 'softmax', 'Xavier'),
]

my_model.build(optimizer='Adam')
# my_model.forprop(np.random.rand(12, 28, 28))
my_model.save('saved models', 'model2')
my_model.load('saved models/model2.npz')
# my_model.forprop(np.random.rand(12, 28, 28))

tracker = MetricTracker(my_model, ['training accuracy', 'training losses', 'gradient magnitude', 'gradient extremes', 'activation magnitude'])

my_model.train(
    data,
    epochs=100,
    batch_size=200,
    learning_rate=0.001,
    clip_value=4,
    tracker=tracker,
    validation_freq=5,
    validation_sample_size=256
)
# my_model.test(data.validation_data, data.validation_labels, 5)
my_model.save('saved models', 'model2')
