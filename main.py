from model import SequentialModel
from layers import Dense, Reshape, Dropout, Convolution, Pooling, Flatten
from data import Dataset
from optimizers import Adam
import numpy as np

data = Dataset.mnist()

my_model = SequentialModel()

my_model.layers = [
    Dense(64, "relu", 'He'),
    Dense(32, "relu", 'He'),
    Dense(10, 'softmax', 'Xavier'),
]

my_model.build(input_shape=(-1, 784),
               optimizer='Adam',
               loss_func="cce",
               tracked_metrics=('training accuracy', 'training losses',
                                'gradient magnitude', 'gradient extremes',
                                'activation magnitude', 'activation extremes')
               )

# my_model.save('saved models', 'model2')
my_model.load('saved models/nano.npz')
# my_model.forprop(np.random.rand(12, 784))
# my_model.backprop(np.random.rand(12, 10), 12, 0, Adam(), 5)

'''my_model.train(
    data,
    epochs=30,
    batch_size=300,
    learning_rate=0.001,
    clip_value=4,
    readout_freq=5,
    readout_sample_size=256
)'''
# my_model.test(data.validation_data, data.validation_labels, 5)
# my_model.save('saved models', 'nano')
