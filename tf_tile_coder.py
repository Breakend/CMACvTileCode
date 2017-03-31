import tensorflow as tf
import numpy as np
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# NOTE: This implementation is based off of a combination of
# https://gist.github.com/bionicv/3703465
# https://github.com/mohammadpz/Theano_Tile_Coding
# and q12.org/phd/thesis/chapter3.pdf
# Some parts may be directly taken from there, but in general everything is heavily modified
# to fit the tensorflow paradigm

class TileCoder(object):
    '''
    Args:
        n_a: the number of association units
        resolution: how many partitions to divide the space into for each dimension
        state_range: tuple of floats (x_min, x_max)
        input_dims: the number of features in the input vectors
        mode: whether to use uniform or assymetrical tiling (to be done in future release)
        learning_rate: the learning rate to increment the weights by
    '''
    def __init__(self, n_a, resolution, state_range, input_dims, mode='uniform', learning_rate=.1):
        # if(mode = 'asm'): # TODO: implement asymmetrical tiling ?
        self.min, self.max = state_range
        self.resolution = resolution
        self.learning_rate = learning_rate
        self.n_a = n_a
        self.input_dims = input_dims

        # weight vectors
        self.weights = tf.Variable(tf.zeros((self.n_a,) + self.input_dims *(self.resolution,)), name="weights")
        # Weight i in the weight table for output x_j,

        # TODO: possible improvement in the future: make weights a hashtable
        # https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/lookup/HashTable?hl=bn
        # these are the offsets into the receptive field association units (tiles)

        self.offsets = tf.range(self.n_a)

    def quantize_and_associate(self, x):
        """
        The quantize and associate function as described in q12.org/phd/thesis/chapter3.pdf
        """
        # for all inputs, bound them by the max values
        q = tf.clip_by_value(x, self.min, self.max)

        # quantize input
        #TODO: possible improvement in the future, can we replace with tf.quantize_v2?
        q = (self.resolution ) * (q - self.min) / (self.max - self.min)

        q = tf.clip_by_value(q, 0.0, self.resolution - 1) # enforce 0 \le q < resolution

        p = tf.add(tf.expand_dims(q, 1), tf.expand_dims(tf.cast(self.offsets, tf.float32), 0)) / self.n_a

        p = tf.transpose(tf.to_int32(p))

        indices = tf.reshape(tf.range(self.n_a), [-1, 1])

        for i in range(self.input_dims):
            indices = tf.concat([indices, tf.reshape(p[:,i], [-1, 1])], axis=1)

        return indices

    def map(self, x):
        """
        The mapping function. Summing the weights of the input neurons
        """
        indices = self.quantize_and_associate(x)
        selected = tf.gather_nd(self.weights, indices)
        y_hat = tf.reduce_sum(selected)
        return y_hat, indices

    def update_rule(self, y, y_hat, learning_rate, indices):
        learning_rate = learning_rate / self.n_a
        delta = tf.SparseTensor(tf.cast(indices,tf.int64), tf.ones(self.resolution), self.weights.get_shape())
        c = tf.zeros(self.weights.get_shape())

        result = c + tf.sparse_tensor_to_dense(delta)

        update = tf.cast(result, tf.float32) * learning_rate * (y - y_hat)
        update_op = tf.assign(self.weights, tf.add(self.weights, update))

        return update_op

    def train(self, dataset, fig, gt_func):
        x_input = tf.placeholder(tf.float32, shape=[self.input_dims])
        y_input = tf.placeholder(tf.float32, shape=[None])
        y_hat, indices = self.map(x_input)
        updates = self.update_rule(y_input, y_hat, 0.1, indices)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        errors = []

        for i, datapoint in enumerate(dataset):
            x, y = datapoint
            x = np.array(x)

            y = np.array(y)
            y = np.reshape(y, (1,))

            preds = sess.run(updates, feed_dict={x_input:x.astype('float32'), y_input:y.astype('float32')})

            eval_ = lambda g : sess.run(y_hat, feed_dict={x_input : g})

            if i <= 20:
                step = 5
            elif i <= 100:
                step = 20
            elif i <= 1500:
                step = 100
            else:
                step = 1000

            if i % 100 == 0:
                errors.append(get_mse(eval_, gt_func))

            if i % step == 0:
                plot_function(fig, eval_, 'Seen points: ' + str(i))

            if i == len(dataset) - 1:
                plot_function(fig, eval_, 'Learned Function', True)

        fig = plt.figure(1)

        plt.plot(errors)
        plt.draw()
        plt.savefig('output_images/mse__cmac.png')
        plt.pause(.0001)
