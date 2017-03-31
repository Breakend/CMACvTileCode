# NOTE: much of this was taken from http://stackoverflow.com/questions/41550966/why-deep-nn-cant-approximate-simple-lnx-function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from utils import *

# Functions from: https://www.benjoffe.com/code/tools/functions3d/examples

f_type = "wave"

def target_function_batch(x):
    # import pdb; pdb.set_trace()
    if f_type == "wave":
        return np.sin(x[:,0]) + np.cos(x[:,1]) + 0.01 * np.random.randn()
    elif f_type == "torus":
        return (0.16 - (0.6 - (x[:,0]**2 + x[:,1])**2))**.5
    elif f_type == "tube":
        return 1.0 / (15.0*(x[:,0]**2 + x[:,1]**2))
    else:
        raise Exception("No such function")

def target_function(x):
    if f_type == "wave":
        return np.sin(x[0]) + np.cos(x[1]) + 0.01 * np.random.randn()
    elif f_type == "torus":
        return (0.16 - (0.6 - (x[0]**2 + x[1])**2))**.5
    elif f_type == "tube":
        return 1.0 / (15.0*(x[0]**2 + x[1]**2))
    else:
        raise Exception("No such function")

def get_mse_mlp(function, gt_function):
    x_0 = np.linspace(0, 7, 100)
    x_1 = np.linspace(0, 7, 100)
    z = np.zeros((100, 100))
    z_g = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            x_in = np.array([x_0[i], x_1[j]])
            z[j, i] = function(np.array([x_in]))
            z_g[j, i] = gt_function(np.array([x_0[i], x_1[j]]))

    return mean_squared_error(z,z_g)


def plot_function_mlp(ax, function, text, hold=False, tag='cmac'):
    #TODO: change this
    ax.cla()
    x_0 = np.linspace(0, 7, 100)
    x_1 = np.linspace(0, 7, 100)
    z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            x_in = np.array([x_0[i], x_1[j]])
            z[j, i] = function(np.array([x_in]))
    X_0, X_1 = np.meshgrid(x_0, x_1)
    ax.plot_surface(X_0, X_1, z, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(X_0, X_1, z, zdir='z', offset=-3, cmap=cm.coolwarm)
    ax.contourf(X_0, X_1, z, zdir='x', offset=-1, cmap=cm.coolwarm)
    ax.contourf(X_0, X_1, z, zdir='y', offset=-1, cmap=cm.coolwarm)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_zlim(-3, 3)
    ax.view_init(45, 45)
    ax.set_title(text)
    if hold:
        plt.show()
    else:
        plt.draw()
        plt.savefig('output_images/' + text + '__' + tag + '.png')
        plt.pause(.0001)


def get_batch(num_samples, min_val, max_val):
    # num_features is assumed to be 2 for this example.
    # However it can take any value in general.
    x = np.random.rand(num_samples, 2).astype(np.float32) * max_val
    y = target_function_batch(x)
    return x, y


def multilayer_perceptron(x, weights, biases):
    """Create model."""
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Parameters
learning_rate = 0.01
training_epochs = 10**4
batch_size = 100
display_step = 500

# Network Parameters
n_hidden_1 = 100  # 1st layer number of features
n_hidden_2 = 50  # 2nd layer number of features
n_hidden_3 = 10  # 2nd layer number of features
n_input = 2


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_3, 1], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden_3])),
    'out': tf.Variable(tf.constant(0.1, shape=[1]))
}

x_data = tf.placeholder(tf.float32, [None, n_input])
y_data = tf.placeholder(tf.float32, [None, 1])

# Construct model
pred = multilayer_perceptron(x_data, weights, biases)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(pred - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train = optimizer.minimize(loss)
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)
errors = []

fig = plt.figure(figsize=np.array([12, 5]))
ax_0 = fig.add_subplot(1, 2, 1, projection='3d')
ax_1 = fig.add_subplot(1, 2, 2, projection='3d')
plot_function(ax_0, target_function, 'Target function', tag='mlp')

for i in range(training_epochs):
    # for i, datapoint in enumerate(dataset):
    x,y  = get_batch(batch_size, 0, 7)
    # x, y = datapoint
    # x = np.array([x])
    y = np.reshape(np.array(y), (batch_size,1))

    # y = np.array(y)
    # y = np.reshape(y, (1,))
    # y_in = get_target_result(x_in)
    sess.run(train, feed_dict={x_data: x, y_data: y})
    eval_ = lambda x_vals : sess.run(pred, feed_dict={x_data: x_vals})

    if i <= 2000:
        step = 200
    elif i <= 10000:
        step = 1000
    elif i <= 50000:
        step = 5000
    else:
        step = 10000

    if i % 200 == 0:
        errors.append(get_mse_mlp(eval_, target_function))

    if i % step == 0:
        plot_function_mlp(ax_1, eval_, 'Seen points: ' + str(i*batch_size), tag='mlp')

plot_function_mlp(ax_1, eval_, 'Learned Function', True, tag='mlp')

fig = plt.figure(1)
# import pdb; pdb.set_trace()
plt.plot(np.arange(len(errors))*batch_size, errors)
plt.draw()
plt.savefig('output_images/mse__mlp.png')
plt.pause(.0001)
