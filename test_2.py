import sys
import os
import tensorflow as tf
import matplotlib
import numpy as np
from simulator import base_classes_GR
from matplotlib import pyplot as plt
import pickle
import helper_functions_gr as hf


AU = 149.6e6 * 1000 # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY # Year
delta_time = (0.5/24.) # 30 minutes
MSUN = 1.9885e+30 # kg
MEARTH = 5.9724e+24 # kg
G = 6.67428e-11/AU**3*MSUN*DAY**2 # Change units of G to AU^3 MSun^{-1} Day^{-2}
c = (2.99792458 * 10**8) * DAY / AU #speed of light in AU/Day

'''
D = np.array(np.ones((10, 3)))
D = D + 1
V = np.array(np.ones((10, 3)))
D_V = np.concatenate([D, V], axis=-1)
D_V = tf.convert_to_tensor(D_V, dtype='float32')

x = hf.cartesian_to_spherical_coordinates(D_V)
print(x.shape)
'''

MMARS = 0.33011 * 10 ** 24 / MSUN

constant = G * MMARS

print(G)
