

import numpy as np
npzfile = np.load('mnist.npz') 
X_train = npzfile['X_train']
X_test = npzfile['X_test']
y_train = npzfile['y_train']
y_test = npzfile['y_test']


