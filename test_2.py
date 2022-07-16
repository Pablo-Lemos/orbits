import sys
import tensorflow as tf
import matplotlib
import numpy as np
from simulator import base_classes_GR
from matplotlib import pyplot as plt
import pickle


x = np.array(np.zeros((10, 3, 3)))
x[:, 1, :] = x[:, 1, :] + [1, 1, 1]
x[:, 2, :] = x[:, 2, :] + [2, 2, 2]

D_tr = np.reshape(x, [1, -1, 3])


#print('x = ',  x[:2, :, :], '\n')

#print('D = ', D_tr[:, :4, :])

#print('D.shape = ', np.shape(D_tr))
print(x.shape[1])

