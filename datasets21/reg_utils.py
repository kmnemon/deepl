import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def load_planar_dataset(seed):
    
    np.random.seed(seed)
    
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])
                    
    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
        assert(parameters['W' + str(l)].shape == layer_dims[l], 1)

        
    return parameters





def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
        
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(i)] = Wi
                    parameters['b' + str(i)] = bi
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(i)] = dWi
                    grads['db' + str(i)] = dbi
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    n = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(n):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    a3, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results

    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def compute_cost(a3, Y):
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.nansum(logprobs)
    
    return cost

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    train_set_x = train_set_x_orig/255
    test_set_x = test_set_x_orig/255

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions

def load_planar_dataset(randomness, seed):
    
    np.random.seed(seed)
    
    m = 50
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 2 # maximum ray of the flower

    for j in range(2):
        
        ix = range(N*j,N*(j+1))
        if j == 0:
            t = np.linspace(j, 4*3.1415*(j+1),N) #+ np.random.randn(N)*randomness # theta
            r = 0.3*np.square(t) + np.random.randn(N)*randomness # radius
        if j == 1:
            t = np.linspace(j, 2*3.1415*(j+1),N) #+ np.random.randn(N)*randomness # theta
            r = 0.2*np.square(t) + np.random.randn(N)*randomness # radius
            
        X[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def load_2D_dataset():

    train_X = np.array([ [-1.58986000e-01,  -3.47926000e-01,  -5.04608000e-01,
        -5.96774000e-01,  -5.18433000e-01,  -2.92627000e-01,
        -1.58986000e-01,  -5.76037000e-02,  -7.14286000e-02,
        -2.97235000e-01,  -4.17051000e-01,  -4.40092000e-01,
        -3.24885000e-01,  -2.46544000e-01,  -2.18894000e-01,
        -3.43318000e-01,  -5.09217000e-01,  -3.84793000e-01,
        -1.49770000e-01,  -1.95853000e-01,  -3.91705000e-02,
        -1.08295000e-01,  -1.86636000e-01,  -2.18894000e-01,
        -8.06452000e-02,   6.68203000e-02,   9.44700000e-02,
         1.86636000e-01,   6.22120000e-02,   2.07373000e-02,
         2.99539000e-02,  -9.90783000e-02,  -6.91244000e-03,
         1.31336000e-01,   2.32719000e-01,   8.52535000e-02,
        -1.31336000e-01,   2.30415000e-03,   1.22120000e-01,
        -3.47926000e-01,  -2.28111000e-01,  -7.60369000e-02,
         4.37788000e-02,   1.15207000e-02,  -4.17051000e-01,
        -3.15668000e-01,   1.26728000e-01,   2.05069000e-01,
         2.18894000e-01,   7.14286000e-02,  -1.31336000e-01,
        -2.09677000e-01,  -2.28111000e-01,  -1.45161000e-01,
        -6.68203000e-02,   1.35945000e-01,   2.69585000e-01,
         2.97235000e-01,   2.74194000e-01,   2.55760000e-01,
         2.23502000e-01,   1.82028000e-01,   1.58986000e-01,
         7.14286000e-02,   1.61290000e-02,  -2.53456000e-02,
        -1.15207000e-02,  -2.30415000e-03,   2.53456000e-02,
         2.53456000e-02,   1.15207000e-02,  -4.83871000e-02,
        -8.52535000e-02,  -9.90783000e-02,  -1.61290000e-02,
         1.31336000e-01,   2.23502000e-01,   2.92627000e-01,
         2.60369000e-01,   2.00461000e-01,   1.72811000e-01,
        -1.31336000e-01,  -1.49770000e-01,  -2.41935000e-01,
        -3.01843000e-01,  -2.97235000e-01,  -2.74194000e-01,
        -3.24885000e-01,  -3.98618000e-01,  -4.35484000e-01,
        -4.72350000e-01,  -3.38710000e-01,  -2.69585000e-01,
        -2.55760000e-01,  -1.68203000e-01,  -1.12903000e-01,
        -3.91705000e-02,  -1.26728000e-01,  -2.32719000e-01,
        -3.38710000e-01,  -4.12442000e-01,  -5.09217000e-01,
        -5.41475000e-01,  -5.04608000e-01,  -4.90783000e-01,
        -3.61751000e-01,  -2.69585000e-01,  -2.23502000e-01,
        -1.86636000e-01,  -1.54378000e-01,  -1.12903000e-01,
        -8.52535000e-02,  -8.52535000e-02,  -1.68203000e-01,
        -1.91244000e-01,  -1.40553000e-01,  -2.99539000e-02,
        -2.00461000e-01,  -1.08295000e-01,   3.45622000e-02,
         8.06452000e-02,  -3.85369000e-01,  -3.81221000e-01,
        -3.52189000e-01,  -3.54263000e-01,  -4.14401000e-01,
        -4.99424000e-01,  -2.98272000e-01,  -3.16935000e-01,
        -3.68779000e-01,  -3.56336000e-01,  -2.71313000e-01,
        -1.77995000e-01,  -2.46429000e-01,  -2.50576000e-01,
        -2.21544000e-01,  -2.15323000e-01,  -1.30300000e-01,
        -2.07028000e-01,  -9.71198000e-02,  -3.90553000e-02,
         1.90092000e-02,  -3.69816000e-02,  -6.39401000e-02,
        -1.30300000e-01,  -3.75000000e-01,  -3.95737000e-01,
        -3.54263000e-01,  -4.37212000e-01,  -4.80760000e-01,
        -4.10253000e-01,  -2.48502000e-01,  -2.27765000e-01,
        -2.83756000e-01,  -2.92051000e-01,  -3.37673000e-01,
        -2.77535000e-01,  -2.07028000e-01,  -1.86290000e-01,
        -1.32373000e-01,  -1.77995000e-01,  -1.65553000e-01,
        -1.61406000e-01,   3.45622000e-04,   7.91475000e-02,
        -2.66129000e-02,  -5.35714000e-02,  -1.41705000e-02,
        -7.01613000e-02,  -6.39401000e-02,  -3.07604000e-02,
        -5.77189000e-02,  -5.35714000e-02,   5.21889000e-02,
        -1.62442000e-02,  -6.39401000e-02,  -6.18664000e-02,
        -3.80184000e-03,   4.18203000e-02,   7.91475000e-02,
         4.59677000e-02,   1.18548000e-01,   1.10253000e-01,
         1.08180000e-01,   1.66244000e-01,   1.41359000e-01,
         1.43433000e-01,   1.70392000e-01,   1.08180000e-01,
         1.18548000e-01,   1.26843000e-01,  -8.67512000e-02,
        -4.73502000e-02,   2.52304000e-02,   6.25576000e-02,
        -5.87558000e-03,  -5.14977000e-02,  -8.05300000e-02,
        -1.53111000e-01,  -1.11636000e-01,  -1.63479000e-01,
        -2.52650000e-01,  -2.46429000e-01,  -3.21083000e-01,
        -3.31452000e-01,  -3.85369000e-01,  -3.99885000e-01,
        -1.24078000e-01,  -3.16935000e-01,  -2.94124000e-01,
        -1.53111000e-01],
        [ 0.423977  ,  0.47076   ,  0.353801  ,  0.114035  , -0.172515  ,
       -0.207602  , -0.0438596 ,  0.143275  ,  0.27193   ,  0.347953  ,
        0.201754  ,  0.00877193, -0.0321637 ,  0.0555556 ,  0.201754  ,
        0.160819  ,  0.0789474 , -0.0906433 ,  0.125731  ,  0.324561  ,
       -0.219298  , -0.30117   , -0.330409  , -0.423977  , -0.564327  ,
       -0.517544  , -0.324561  , -0.166667  , -0.0730994 , -0.195906  ,
       -0.342105  , -0.377193  , -0.464912  , -0.429825  , -0.195906  ,
       -0.0847953 , -0.236842  , -0.125731  , -0.00292398, -0.312865  ,
       -0.125731  ,  0.0146199 ,  0.0204678 ,  0.154971  , -0.160819  ,
       -0.318713  , -0.219298  , -0.312865  , -0.459064  , -0.646199  ,
       -0.605263  , -0.581871  , -0.429825  , -0.412281  , -0.482456  ,
       -0.511696  , -0.406433  , -0.295322  , -0.172515  , -0.0497076 ,
       -0.0497076 , -0.0847953 , -0.154971  , -0.21345   , -0.266082  ,
       -0.383041  , -0.482456  , -0.505848  , -0.511696  , -0.55848   ,
       -0.657895  , -0.646199  , -0.552632  , -0.5       , -0.423977  ,
       -0.359649  , -0.371345  , -0.30117   , -0.207602  , -0.225146  ,
       -0.27193   ,  0.0906433 ,  0.0730994 ,  0.0614035 ,  0.178363  ,
        0.195906  ,  0.307018  ,  0.295322  ,  0.266082  ,  0.160819  ,
        0.0789474 ,  0.0438596 ,  0.0438596 ,  0.102339  ,  0.266082  ,
        0.30117   ,  0.347953  ,  0.44152   ,  0.44152   ,  0.418129  ,
        0.353801  ,  0.219298  ,  0.0146199 , -0.125731  , -0.143275  ,
       -0.137427  , -0.0847953 , -0.0789474 , -0.0380117 , -0.00877193,
        0.0555556 ,  0.137427  ,  0.277778  ,  0.30117   ,  0.195906  ,
       -0.0497076 ,  0.0672515 , -0.230994  , -0.0847953 ,  0.0672515 ,
        0.119883  ,  0.0330409 ,  0.131287  ,  0.258187  ,  0.36462   ,
       -0.0692982 , -0.0324561 , -0.0979532 , -0.183918  , -0.290351  ,
       -0.396784  ,  0.00438596,  0.0862573 ,  0.143567  ,  0.229532  ,
        0.376901  ,  0.295029  ,  0.217251  ,  0.0289474 ,  0.213158  ,
        0.258187  ,  0.401462  ,  0.42193   ,  0.331871  ,  0.376901  ,
       -0.437719  , -0.351754  , -0.20848   , -0.376316  , -0.503216  ,
       -0.466374  , -0.257602  , -0.314912  , -0.384503  , -0.454094  ,
       -0.519591  , -0.548246  , -0.535965  , -0.478655  , -0.50731   ,
       -0.298538  , -0.175731  , -0.126608  ,  0.258187  ,  0.356433  ,
        0.180409  ,  0.0780702 , -0.052924  , -0.16345   , -0.294444  ,
       -0.466374  , -0.527778  , -0.396784  , -0.417251  , -0.167544  ,
       -0.0856725 , -0.0160819 ,  0.00438596,  0.204971  ,  0.19269   ,
        0.254094  ,  0.19269   ,  0.0862573 , -0.0692982 , -0.024269  ,
        0.0657895 ,  0.168129  ,  0.19269   ,  0.299123  ,  0.319591  ,
        0.393275  ,  0.42193   ,  0.507895  ,  0.520175  ,  0.552924  ,
        0.442398  ,  0.573392  ,  0.507895  ,  0.552924  ,  0.54883   ,
        0.49152   , -0.188012  , -0.0365497 , -0.433626  , -0.605556  ,
       -0.515497  , -0.62193   , -0.126608  , -0.228947  , -0.134795  ,
        0.184503  ]])



    train_Y = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 0, 0]])


    test_X = np.array([[-0.35306235, -0.2271258 ,  0.09289767,  0.14824252, -0.00151249,
        0.04533501,  0.14434613, -0.13807064, -0.16385952, -0.19535742,
        0.1863405 , -0.49340321, -0.0918527 , -0.10428124, -0.20888058,
        0.24239325,  0.14986252, -0.49949386,  0.14708703, -0.20265487,
        0.09335189,  0.11339545,  0.25152129, -0.07166737,  0.18055064,
       -0.18110124,  0.27785389, -0.01327092, -0.01502107, -0.00308691,
        0.2141149 ,  0.04526842, -0.13417062, -0.05519283, -0.45642962,
       -0.44930938, -0.33460047,  0.14155288, -0.24788379, -0.04867248,
       -0.19644304, -0.49903366, -0.04108153, -0.28101764, -0.1756489 ,
       -0.46788726, -0.41606452,  0.29091938,  0.02219132, -0.42158567,
        0.1312701 , -0.41008228, -0.35553717, -0.35760857, -0.30571571,
       -0.02607286, -0.11757221, -0.24149786, -0.18033413,  0.26392833,
       -0.32814337, -0.29850456, -0.11643577, -0.46780346,  0.06911927,
       -0.04271503, -0.3385095 , -0.16095543, -0.07511389, -0.42933428,
       -0.16383887, -0.45116398,  0.0477326 , -0.20718875,  0.12999743,
       -0.17031431,  0.16669776,  0.10966868,  0.02952713, -0.38108519,
       -0.03730684,  0.23863706,  0.1794064 , -0.44541368, -0.13173944,
        0.25816653, -0.17144475,  0.0223521 ,  0.2175178 ,  0.16482247,
        0.10561683, -0.4963785 , -0.2907698 , -0.35518369, -0.11756326,
        0.2124443 , -0.33588692, -0.21834424,  0.2664948 ,  0.01073067,
        0.14320849, -0.49639587,  0.01985078, -0.49432988, -0.10664757,
       -0.00319048, -0.42560461,  0.01205282, -0.42333424, -0.18530875,
       -0.21421786, -0.31768577,  0.14291722, -0.32157713, -0.08101309,
       -0.25938858, -0.01728897, -0.00865689,  0.0123751 ,  0.2679647 ,
       -0.2220084 ,  0.17236739,  0.13126105,  0.24179202,  0.1222105 ,
        0.05346095, -0.2122828 , -0.40454458, -0.37627313, -0.40516269,
       -0.3576407 , -0.05349795, -0.29150432,  0.20878861, -0.02555617,
       -0.44039164,  0.19021425,  0.12623856, -0.33099112, -0.31851586,
       -0.16257604,  0.00302448, -0.09090669,  0.16334486,  0.11745137,
        0.05169295, -0.43946652,  0.20343101,  0.24735643, -0.28638208,
        0.03336897, -0.22278369, -0.27930754,  0.10390813, -0.22047998,
       -0.22122989,  0.10595997,  0.00536441, -0.18630331,  0.07871049,
       -0.03904464,  0.02897219, -0.24701344,  0.26144865, -0.1457531 ,
        0.0298899 , -0.20212221,  0.13433263, -0.40741669, -0.40943778,
        0.25701656,  0.11003269,  0.25896689,  0.15383051, -0.14735839,
        0.14736828, -0.03874009,  0.2255192 ,  0.26950824,  0.12482879,
        0.17162781,  0.005052  , -0.35458825, -0.47738659, -0.24116903,
        0.09375793, -0.29988892, -0.16247809,  0.19682557, -0.34801539,
        0.14035793, -0.04634861, -0.23929465, -0.45785597, -0.10819348,
        0.00520265,  0.17635156,  0.127651  ,  0.24868221, -0.31689909],

        [-0.67390181,  0.44731976, -0.75352419, -0.71847308,  0.16292786,
        0.20982573, -0.68720754,  0.55241732,  0.11134314,  0.4011457 ,
        0.52186234,  0.06007882, -0.12178156, -0.54557988,  0.15980217,
       -0.44476841, -0.10482911,  0.01644946,  0.17669485, -0.18450312,
       -0.08481748,  0.20626961, -0.24962124,  0.38027844, -0.49845242,
       -0.7543908 ,  0.24844372, -0.48747145,  0.36967342,  0.21570428,
        0.4517632 , -0.62350648, -0.75042241, -0.08569443,  0.30457343,
        0.2521395 ,  0.28108967, -0.68735638, -0.44652464, -0.79481614,
       -0.60949512, -0.48378051, -0.33717187, -0.02116391, -0.16079769,
       -0.03606064,  0.50923811,  0.00281817,  0.10162396,  0.39027243,
       -0.19991088, -0.67280407,  0.27731659,  0.52170388,  0.38303523,
        0.37125026, -0.02881462, -0.04582353,  0.00628885, -0.19920115,
       -0.62828486, -0.71993032, -0.41548515, -0.11866615, -0.51905734,
        0.33006558,  0.07544364,  0.15532539,  0.52723462, -0.32823455,
        0.59636143, -0.36885502,  0.08964546,  0.029608  , -0.76144357,
        0.26024428, -0.66180432, -0.02214623,  0.1205507 , -0.68066679,
       -0.16967073, -0.33979185,  0.36614806,  0.48773358, -0.37301885,
        0.3214765 , -0.05436293,  0.46044783, -0.64838015, -0.56009307,
       -0.47356917, -0.53039871,  0.39282998, -0.19033134,  0.45045063,
        0.24409336, -0.42208135,  0.42598976, -0.41708775,  0.1157563 ,
        0.57290223,  0.24764736, -0.3142498 ,  0.40264768, -0.00867353,
        0.50973211,  0.21494326,  0.43847113, -0.22920959, -0.21966031,
        0.57175744, -0.34148012,  0.08564325,  0.15996528,  0.01944172,
       -0.05106505, -0.57578078,  0.40091322, -0.27619353, -0.49294225,
       -0.28870081, -0.47234967, -0.23841632, -0.23915142, -0.18139411,
       -0.05415873, -0.49059782, -0.53062928, -0.36278139,  0.00920269,
       -0.18299568, -0.53677784, -0.03586948, -0.21603988,  0.26925529,
       -0.7207657 , -0.7314299 , -0.4287631 , -0.01338599,  0.06686597,
       -0.72654767, -0.71336776, -0.48357423,  0.29270667, -0.10436159,
       -0.49777696,  0.24581507,  0.00870498,  0.06879417,  0.12963342,
       -0.67034231, -0.14869216,  0.43238638,  0.10306382, -0.76702369,
        0.02037055, -0.12451613,  0.10208984,  0.08037445,  0.04847465,
        0.18299267, -0.02251376,  0.35259805,  0.38051718, -0.46868559,
        0.10126281, -0.3266953 , -0.63985405,  0.55315944,  0.46177788,
       -0.52299972, -0.52541811, -0.45826105,  0.10749309,  0.44665931,
       -0.41425849, -0.48927081,  0.27762013, -0.66565116,  0.50814405,
       -0.42429453, -0.04760931,  0.37126309, -0.68693223,  0.28161654,
       -0.77221995, -0.67487155,  0.17209416, -0.14566765, -0.2592839 ,
       -0.68428925, -0.14915378,  0.56220974,  0.25374027,  0.59753216,
       -0.54444942, -0.57245363, -0.34093757, -0.49750183, -0.42941273]]
                                                                                )
    test_Y = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1,
       0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
       0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1,
       1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]])


#    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);

    return train_X, train_Y, test_X, test_Y