from model import SequentialModel
from layers import Dense, Flatten
from data import Dataset


training_data, training_labels, validation_data, validation_labels = Dataset.mnist()

my_model = SequentialModel()

my_model.layers = [
    Flatten(input_shape=[-1, 28, 28]),
    Dense(128, "relu", 'He'),
    Dense(64, "relu", 'He'),
    Dense(10, 'softmax', 'Xavier'),
]

my_model.build(optimizer='Adam')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))
my_model.save('saved models', 'model1')
my_model.load('saved models/model1.npz')
# my_model.forprop(np.random.rand(12, 28, 28))
# my_model.backprop(np.random.rand(10, 12))

my_model.train(
    training_data, training_labels,
    epochs=2,
    batch_size=256,
    learning_rate=0.01,
    clip_value=2
)

my_model.save('saved models', 'model1')
