#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np
import random

_maxLongintBy4 = _maxLongint // 4       # maximum integer divided by 4
_randomTable = [random.randrange(_maxLongintBy4) for i in xrange(2048)]   #table of random numbers

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
    def __init__(self, n_tiling, n_partition, state_range, mode='uniform')
        # if(mode = 'asm'): # TODO: implement asymmetrical tiling ?
        self.min, self.max = state_range
        self.n_partition = n_partition
        self.width = float(self.max-self.min)/self.n_partition

        # Now create n_tiling n_partition x n_partition tilings
        self.tiles = tf.placeholder(tf.float32, shape=[self.n_tiling, self.n_partition, self.n_partition])


    def eval(self, input)
        # TODO:
        return val


    def tile(self, x):
        offsets = self.n_partition * range(self.n_tiling) / self.self.n_tiling


def main():


if __name__ == '__main__':
    main()
