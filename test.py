import numpy as np

x = np.random.randint(0, 10, (5000, 4000))
np.savez_compressed('large_arrays.npz', array1=x, array2='waht the sigma')

# Load the arrays
loaded_data = np.load('large_arrays.npz')
loaded_array1 = loaded_data['array1']
loaded_array2 = loaded_data['array2']
print(loaded_array2)