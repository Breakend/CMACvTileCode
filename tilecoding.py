#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np
import random

# void tiles(
#  int tiles[],     // provided array will contain the returned tile indices
#  int num_tilings, // number of tile indices to be returned
#  int memory_size, // total number of possible tiles
#  float floats[],  // array of floating-point vars to be converted to tiles
#  int num_floats,  // number of floating-point variables
#  int ints[],      // array of integer variables to be converted to tiles
#  int num_ints)


class Tiling(object):
    '''
    Args:
        n_tiling: dimension of the state space
        n_partition: how many partitions to divide the space into for each dimension
        state_range: tuple of floats (x_min, y_min)
    '''
    def __init__(self, n_tiling, n_partition, state_range, mode='uniform'):
        # if(mode = 'asm'): # TODO: implement asymmetrical tiling ?
        self.min, self.max = state_range
        self.n_partition = n_partition
        self.width = tf.Variable(float(self.max-self.min)/self.n_partition, dtype=tf.float64)
        self.n_tiling = n_tiling

        # Now create n_tiling n_partition x n_partition tilings
        self.tiles = tf.placeholder(tf.float64, shape=[self.n_tiling, self.n_partition, self.n_partition])


    def eval(self, input):
        # TODO:
        return val


    def tile(self, x):
        offsets = self.n_partition * tf.range(self.n_tiling) / self.n_tiling
        # mapped_x = x.dimshuffle(0, 'x') + offsets.dimshuffle('x', 0)
        mapped_x = tf.add(tf.expand_dims(x, 1), tf.expand_dims(tf.cast(offsets, tf.float64), 1))

        # Since we have an extra tile:
        max_val = self.max + self.width

        # Mapping into range [0, num_tiles + 1] to be able to do quantization
        # by casting to int.
        mapped_x = ((self.n_partition + 1) * mapped_x /
                    (max_val - self.min))
        # shape: (n_tiling, n_partition, n_partition)
        
        # Update tiles
        tiles = mapped_x
        q_x = tf.to_int32(tf.transpose(mapped_x))
        return q_x


def main():
    test = Tiling(3.0, 10.0, (0.0, 10.0))

    q_test = test.tile(np.zeros((10, 10)))
    print np.shape(q_test)

if __name__ == '__main__':
    main()
