import numpy as np
import matplotlib.pyplot as plt
from datasets21.reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from datasets21.reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
#from testCases import *

#plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

#train_X, train_Y, test_X, test_Y = load_2D_dataset()
load_2D_dataset()