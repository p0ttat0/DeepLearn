from model import SequentialModel
from layers import Dense, Reshape, Dropout, Convolution
from data import Dataset
from tracking import MetricTracker
import numpy as np

data = Dataset.mnist()

my_model = SequentialModel()

my_model.layers = [
    Reshape([-1, 784]),
    Dense(64, "swish", 'He'),
    Dense(10, 'softmax', 'Xavier'),
]

# my_model.build(optimizer='Adam')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.save('saved models', 'model2')
# my_model.forprop(np.random.rand(12, 28, 28))

# tracker = MetricTracker(my_model, ['training accuracy', 'training losses', 'gradient magnitude', 'gradient extremes', 'activation magnitude', 'activation extremes'])

'''my_model.train(
    data,
    epochs=10,
    batch_size=300,
    learning_rate=0.001,
    clip_value=4,
    tracker=tracker,
    readout_freq=5,
    readout_sample_size=256
)'''
# my_model.test(data.validation_data, data.validation_labels, 5)
# my_model.save('saved models', 'model2')

x = Convolution(3)
d = x.conv2d_gemm(data.training_data[0].reshape((1, 28, 28, 1)), np.ones((3, 3, 1, 1)), np.zeros(1), padding='same')
import matplotlib.pyplot as plt
plt.imshow(data.training_data[0], cmap='viridis')
plt.show()
plt.imshow(d.reshape((28, 28)), cmap='viridis')
plt.show()
