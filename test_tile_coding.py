import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tf_tile_coder import TileCoder
from utils import plot_function

# max_val = 0

def target_function(x):
    return np.sin(x[0]) + np.cos(x[1]) + 0.01 * np.random.randn()

def get_dataset(num_samples, min_val, max_val):
    # num_features is assumed to be 2 for this example.
    # However it can take any value in general.
    dataset = []
    for i in range(num_samples):
        x = np.array([np.random.random() * max_val, np.random.random() * max_val])
        y = target_function(x)
        dataset.append((x, y))
    return dataset


tile_coder = TileCoder(
    state_range= (0, 7.0), resolution=50,
    n_a=50, input_dims=2,
    learning_rate=0.1)

dataset = get_dataset(10000, 0, 10)

fig = plt.figure(figsize=np.array([12, 5]))
ax_0 = fig.add_subplot(1, 2, 1, projection='3d')
ax_1 = fig.add_subplot(1, 2, 2, projection='3d')
plot_function(ax_0, target_function, 'Target function')
# import ipdb; ipdb.set_trace()

print 'Training'

tile_coder.train(dataset, ax_1, target_function)
