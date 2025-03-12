from model import SequentialModel
from layers import Dense
import numpy as np

my_model = SequentialModel(10)
my_model.layers = [
    Dense(30),
    Dense(40),
    Dense(40),
    Dense(10, activation_function='softmax'),
]

my_model.build()
my_model.save('saved models', 'model')
my_model.load('saved models/model.npz')
print(my_model.forward(np.random.rand(10, 4)))

