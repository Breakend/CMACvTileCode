import tensorflow as tf
import numpy as np
from utils import plot_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class TileCoder(object):
    '''
    Args:
        n_tiling: dimension of the state space
        resolution: how many partitions to divide the space into for each dimension
        state_range: tuple of floats (x_min, y_min)
    '''
    def __init__(self, n_tiling, resolution, state_range, dim, mode='uniform', learning_rate=.1):
        # if(mode = 'asm'): # TODO: implement asymmetrical tiling ?
        self.min, self.max = state_range
        self.resolution = resolution
        self.learning_rate = learning_rate

        # This is the width of the receptive field
        self.width = np.float32(self.max-self.min)/self.resolution
        self.n_tiling = n_tiling
        self.dim = dim

        # n_tiling resolution x resolution tilings
        self.tiles = tf.Variable(tf.zeros([self.n_tiling + self.resolution * self.dim]), name="weights")

        # these are the offsets into the receptive field association units (tiles)
        self.offsets = self.width * tf.range(self.n_tiling) / self.n_tiling

    def quantize_and_associate(self, x):
        # for all inputs, bound them by the max values
        # mapped_x = tf.clip_by_value(x, self.min, self.max)

        # quantize input
        q = (self.resolution + 1) * (x - self.min) / (self.max - self.min)

        q = tf.clip_by_value(q, 0.0, self.resolution) # enforce 0 \le q < resolution

        p = tf.add(tf.expand_dims(x, 1), tf.expand_dims (tf.cast(self.offsets, tf.float32), 0)) / self.n_tiling
        p = tf.to_int32(tf.transpose(p))

        indices = tf.range(self.n_tiling)

        # TODO: why is this a thing?
        for i in range(self.dim):
            indices += [p[:,i]]
        return indices
        # TODO: hashing

    def map(self, x):
        indices = self.quantize_and_associate(x)
        selected = tf.gather(self.tiles, tf.squeeze(indices))
        y_hat = tf.reduce_sum(selected)
        return y_hat

    def update_rule(self, y, y_hat, learning_rate):
        # grad is only used to locate the weights which
        # are used for computing y_hat. Since
        # "y_hat = T.sum(selected)", T.grad will returns
        # 1 for corresponding weights and 0 for others.
        gradients = tf.gradients(y_hat, self.tiles)
        learning_rate = learning_rate / self.n_tiling

        update = learning_rate * (y - y_hat)
        update_op = tf.assign(self.tiles, tf.add(self.tiles, update))
        return update_op
        # batch_updates = [(self.tiles, self.tiles + update)]
        # return batch_updates

    def train(self, dataset, fig):
        x_input = tf.placeholder(tf.float32, shape=[self.dim])
        y_input = tf.placeholder(tf.float32, shape=[None])
        y_hat = self.map(x_input)
        updates = self.update_rule(y_input, y_hat, 0.1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i, datapoint in enumerate(dataset):
            x, y = datapoint
            x = np.array(x)
            # x = np.reshape(x, (-1, 2))
            y = np.array(y)
            y = np.reshape(y, (1,))
            # import pdb; pdb.set_trace()
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
            if i % step == 0:
                plot_function(fig, eval_, 'Seen points: ' + str(i))

            if i == len(dataset) - 1:
                plot_function(fig, eval_, 'Learned Function', True)
