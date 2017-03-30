import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_function(ax, function, text, hold=False):
    ax.cla()
    x_0 = np.linspace(0, 7, 100)
    x_1 = np.linspace(0, 7, 100)
    z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            z[j, i] = function(np.array([x_0[i], x_1[j]]))
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
        plt.savefig(text + '.png')
        plt.pause(.0001)
