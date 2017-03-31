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
        n_a: the number of association units
        resolution: how many partitions to divide the space into for each dimension
        state_range: tuple of floats (x_min, x_max), that is the
    '''
    def __init__(self, n_a, resolution, state_range, input_dims, mode='uniform', learning_rate=.1):
        # if(mode = 'asm'): # TODO: implement asymmetrical tiling ?
        self.min, self.max = state_range
        self.resolution = resolution
        self.learning_rate = learning_rate

        # This is the width of the receptive field?
        #TODO: do we need this...
        self.width = np.float32(self.max-self.min)/self.resolution
        self.n_a = n_a
        self.input_dims = input_dims

        # n_a resolution x resolution tilings
        self.weights = tf.Variable(tf.zeros((self.n_a,) + self.input_dims *(self.resolution,)), name="weights")
        # Weight i in the weight table for output x_j,

        # TODO: make weights a hashtable
        # https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/lookup/HashTable?hl=bn
        # these are the offsets into the receptive field association units (tiles)
        # self.offsets = self.num_dims * tf.range(self.input_dims)
        self.offsets = tf.range(self.n_a)
        # self.offsets = tf.tile(self.offsets, [self.input_dims])
        # self.offsets = tf.reshape(self.offsets, [self.input_dims, self.n_a])
        # should be tensor of size input_dims X n_a,
        # that is n_a offsets per dimension of the input ranging from 0 to n_a
        # import pdb; pdb.set_trace()

    def quantize_and_associate(self, x):
        # for all inputs, bound them by the max values
        q = tf.clip_by_value(x, self.min, self.max)

        # quantize input
        #TODO: can we replace with tf.quantize_v2
        q = (self.resolution ) * (q - self.min) / (self.max - self.min)

        q = tf.clip_by_value(q, 0.0, self.resolution - 1) # enforce 0 \le q < resolution

        # p = tf.add(tf.tile(q, []), tf.expand_dims (tf.cast(self.offsets, tf.float32), 0)) / self.n_a
        # p = tf.expand_dims(q, 1) + tf.cast(self.offsets, tf.float32) / self.n_a
        p = tf.add(tf.expand_dims(q, 1), tf.expand_dims(tf.cast(self.offsets, tf.float32), 0)) / self.n_a



        p = tf.transpose(tf.to_int32(p))

        # indices = tf.tile(tf.reshape(p, [-1]), [self.n_a])
        # indices = tf.reshape(indices, (self.n_a, self.input_dims, self.resolution))
        import pdb; pdb.set_trace()
        # indices = tf.range(self.resolution)
        # indices = tf.tile(indices, [self.input_dims])
        # indices = tf.reshape(indices, [self.input_dims, self.n_a])
        # indices = tf.concat([indices, p], axis=0)
        indices = tf.reshape(tf.range(self.n_a), [-1, 1])


        # import pdb; pdb.set_trace()

        # indices += 1

        for i in range(self.input_dims):
            indices = tf.concat([indices, tf.reshape(p[:,i], [-1, 1])], axis=1)

        # TODO: why is this a thing?
        # for i in range(self.input_dims):
            # indices +=
        # indices
        return indices
        # TODO: hashing

    def map(self, x):
        indices = self.quantize_and_associate(x)
        selected = tf.gather_nd(self.weights, indices)
        y_hat = tf.reduce_sum(selected)
        return y_hat, indices

    def update_rule(self, y, y_hat, learning_rate, indices):
        # grad is only used to locate the weights which
        # are used for computing y_hat. Since
        # "y_hat = T.sum(selected)", T.grad will returns
        # 1 for corresponding weights and 0 for others.
        # gradients = tf.gradients(y_hat, self.weights)
        # instead of this can select the weights to update by multiplying
        # the update by ones where those indices are
        learning_rate = learning_rate / self.n_a
        delta = tf.SparseTensor(tf.cast(indices,tf.int64), tf.ones(self.resolution), self.weights.get_shape())
        c = tf.zeros(self.weights.get_shape())

        result = c + tf.sparse_tensor_to_dense(delta)

        update = tf.cast(result, tf.float32) * learning_rate * (y - y_hat)
        update_op = tf.assign(self.weights, tf.add(self.weights, update))
        # update_op = tf.scatter_nd(indices, update)
        return update_op
        # batch_updates = [(self.weights, self.weights + update)]
        # return batch_updates

    def train(self, dataset, fig):
        x_input = tf.placeholder(tf.float32, shape=[self.input_dims])
        y_input = tf.placeholder(tf.float32, shape=[None])
        y_hat, indices = self.map(x_input)
        updates = self.update_rule(y_input, y_hat, 0.1, indices)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i, datapoint in enumerate(dataset):
            x, y = datapoint
            x = np.array(x)

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
