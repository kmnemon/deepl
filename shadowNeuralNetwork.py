import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X, Y = load_planar_dataset()

shape_X = X.shape
shape_Y = X.shape
#m = shape_X[1]


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer*

    
    """

    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    '''
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    '''

    W1 = np.random.randn(n_h, n_x)
    W2 = np.random.randn(n_y, n_h)
    b1 = np.zeros([n_h, 1])
    b2 = np.zeros([n_y, 1])

    params = {"W1": W1, "W2": W2, "b1": b1, "b2":b2}

    return params

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    trainSize = X.shape[1]


    Z1 =np.dot(W1, X) + b1
    A1 =np.tanh(Z1)

    assert(A1.shape == (W1.shape[0], trainSize))

    Z2 =np.dot(W2, A1) + b2
    A2 =sigmoid(Z2)

    assert(A2.shape == (W2.shape[0], trainSize))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]


    L1 = np.dot(Y, np.log(A2).T)
    L2 = np.dot((1-Y), np.log(1-A2).T)
    J = (-1/m) *(L1 + L2)


    cost = np.squeeze(J)

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """

    m = Y.shape[1]

    A1 = cache["A1"]
    A2 = cache["A2"]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot( dZ2, A1.T) / m
    db2 = 1/m * (np.sum(dZ2, axis=1, keepdims=True))

    dZ1 = np.multiply( np.dot( W2.T, dZ2), (1 - np.power(A1, 2) ) )
    dW1 = np.dot( dZ1, X.T) /m
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1
    }

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 =
















