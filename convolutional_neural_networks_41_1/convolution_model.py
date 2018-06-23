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
    X_pad = np.pad(X, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (0,0))
    ### END CODE HERE ###

    return X_pad

'''
a = np.array(
[
    [1,1,1,1,1,1],
    [2,2,2,2,2,2],
    [3,3,3,3,3,3],
    [4,4,4,4,4,4],
    [5,5,5,5,5,5]
],
[
    [1,1,1,1,1],
    [2,2,2,2,2]
]
)
'''
a = np.random.randn( 2, 2, 2, 2 , 2 )



x = zero_pad(a, 1)
print(x.shape)
print(x)
