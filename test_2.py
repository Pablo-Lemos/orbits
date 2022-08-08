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


def GR_correctoin (m1, m2, distance, velocity):
    """
    Calculates GR correction
    Args:
        m1: mass of first body
        m2: mass of second body
        distance: three dimensional distance array
        velocity: three dimensional velocity array

    Returns: A numpy array with the three force correction components

    """
    dist_norm = np.sum(distance ** 2.) ** 0.5
    L_norm = np.linalg.norm(np.cross(distance, velocity))
    f_n = G * m1 * m2 * distance / dist_norm ** 3.
    corr = 3*L_norm**2/(c**2 * dist_norm**2)
    GR = f_n * (1 + 10000 * corr)

    return f_n


'''
D = np.array(np.ones((10, 3)))
D = D + 1
V = np.array(np.ones((10, 3)))
D_V = np.concatenate([D, V], axis=-1)
D_V = tf.convert_to_tensor(D_V, dtype='float32')

x = hf.cartesian_to_spherical_coordinates(D_V)
print(x.shape)
'''

pos_sun = np.array([7.93567917e-03, -6.29360340e-04, -2.31793679e-04])
vel_sun = np.array([3.56426004e-06, 7.70848450e-06, -1.38462510e-07])

pos_mercury = np.array([-5.78670715e-02, -4.61182491e-01, -3.17988125e-02])  # AU
vel_mercury = np.array([2.22124712e-02, -2.53545004e-03, -2.24740703e-03])  # AU/DAY

MMARS = 0.33011 * 10 ** 24 / MSUN

GR = np.linalg.norm(GR_correctoin(MSUN, MMARS, pos_mercury, vel_mercury))

a = GR / MMARS
print(GR)
