import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tf_tile_coder import TileCoder
from utils import *

# NOTE: This implementation is based off of a combination of
# https://gist.github.com/bionicv/3703465
# https://github.com/mohammadpz/Theano_Tile_Coding
# and q12.org/phd/thesis/chapter3.pdf
# Some parts may be directly taken from there, but in general everything is heavily modified
# to fit the tensorflow paradigm

f_type = "cone"

def target_function(x):
    if f_type == "wave":
        return np.sin(x[0]) + np.cos(x[1]) + 0.01 * np.random.randn()
    elif f_type == "cone":
        xs = np.power(x[0]*2.-4, 2)
        ys = np.power(x[1]*2.-4, 2)
        return np.sin(np.sqrt(xs + ys))
    else:
        raise Exception("No such function")

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
    state_range= (0, 7.0), resolution=100,
    n_a=100, input_dims=2,
    learning_rate=0.1)

dataset = get_dataset(20000, 0, 7)

fig = plt.figure(figsize=np.array([12, 5]))
ax_0 = fig.add_subplot(1, 2, 1, projection='3d')
ax_1 = fig.add_subplot(1, 2, 2, projection='3d')
plot_function(ax_0, target_function, 'Target function')

print 'Training'

tile_coder.train(dataset, ax_1, target_function)
