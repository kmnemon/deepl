import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from Optimization_Methods_22.optimizaiton.opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from Optimization_Methods_22.optimizaiton.opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from Optimization_Methods_22.optimizaiton.testCases import *

#https://blog.csdn.net/koala_tree/article/details/78216371

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# GRADED FUNCTION: update_parameters_with_gd
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
      Update parameters using one step of gradient descent

      Arguments:
      parameters -- python dictionary containing your parameters to be updated:
                      parameters['W' + str(l)] = Wl
                      parameters['b' + str(l)] = bl
      grads -- python dictionary containing your gradients to update each parameters:
                      grads['dW' + str(l)] = dWl
                      grads['db' + str(l)] = dbl
      learning_rate -- the learning rate, scalar.

      Returns:
      parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

    return parameters
