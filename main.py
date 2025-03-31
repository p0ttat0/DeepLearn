from model import SequentialModel
from layers import Dense, Flatten
from data import Dataset
from tracking import MetricTracker

data = Dataset.mnist()

my_model = SequentialModel()

my_model.layers = [
    Flatten(input_shape=[-1, 28, 28], output_shape=[784, -1]),
    Dense(128, "relu", 'He'),
    Dense(64, "relu", 'He'),
    Dense(10, 'softmax', 'He'),
]

my_model.build(optimizer='Adam')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))
# my_model.save('saved models', 'model1')
# my_model.load('saved models/model1.npz')
# my_model.forprop(np.random.rand(12, 28, 28))

monitor = MetricTracker(my_model, ['training accuracy', 'training losses', 'gradient magnitude', 'gradient extremes', 'activation magnitude'])

my_model.train(
    data,
    epochs=10,
    batch_size=128,
    learning_rate=0.01,
    clip_value=10,
    tracker=monitor
)

# my_model.save('saved models', 'model1')
