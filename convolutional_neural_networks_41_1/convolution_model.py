import numpy as np
import h5py
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



np.random.seed(1)
'''
Convolution functions, including:
Zero Padding
Convolve window
Convolution forward
Convolution backward (optional)
Pooling functions, including:
Pooling forward
Create mask
Distribute value
Pooling backward (optional)
'''

#Convolutional Neural Networks

#helper function
#zero-padding
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0,0), (pad, pad), (pad, pad) , (0,0)), 'constant', constant_values = (0,0))
    #X_pad = np.pad(X, ((1, 1), (0, 0), (3, 3)), 'constant', constant_values=(0, 0))
    ### END CODE HERE ###

    return X_pad


# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)

    # Sum over all entries of the volume s.
    Z = np.sum(s)

    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)

    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    #A_prev'shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    #W'shape
    (f, f, _ , n_C) = W.shape

    #hyper-parameter
    pad = hparameters['pad']
    s = hparameters['stride']

    #output size
    n_H = int( (n_H_prev + 2 * pad - f) / s + 1 )
    n_W = int( (n_W_prev + 2 * pad - f) / s + 1 )

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        for c in range(n_C):  #filters
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = s * h
                    vert_end = vert_start + f
                    horiz_start = s * w
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


# GRADED FUNCTION: pool_forward
def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    n_C = n_C_prev

    f = hparameters['f']
    stride = hparameters['stride']

    #output size
    n_H = int( (n_H_prev - f) / stride ) + 1
    n_W = int( (n_W_prev - f) / stride ) + 1

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                vert_start = s * h
                vert_end = vert_start + f
                horiz_start = s * w
                horiz_end = horiz_start + f
                a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                if mode == 'max':








