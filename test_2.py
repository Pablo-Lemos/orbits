import sys
import os
import tensorflow as tf
import matplotlib
import numpy as np
from simulator import base_classes_GR
from matplotlib import pyplot as plt
import pickle
import helper_functions_gr as hf


D = np.array(np.ones((10, 3)))
D = D + 1
V = np.array(np.ones((10, 3)))
D_V = np.concatenate([D, V], axis=-1)
D_V = tf.convert_to_tensor(D_V, dtype='float32')

x = hf.cartesian_to_spherical_coordinates(D_V)
print(x.shape)

