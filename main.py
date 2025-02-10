from sequential import Sequential
from layers import Dense
import numpy as np

my_model = Sequential()
my_model.layers = [
    Dense(30),
    Dense(40),
    Dense(40),
    Dense(10, activation_function='softmax'),
]

my_model.compile(10)
print(my_model.forward(np.random.randint(0, 10, (10, 1))))
