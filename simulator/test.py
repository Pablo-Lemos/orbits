import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pysr
#Define constants
AU = 149.6e9    # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY
MSUN = 1.98892 * 10**30 # Solar mass
MEARTH = 5.9742 * 10**24 # Earth mass
G = 6.67428e-11/AU**3*MSUN*DAY**2 # The gravitational constant G in AU**3 /MSUN/ Day^2
c = (2.99792458 * 10**8) * DAY / AU #speed of light in AU/Day


def GR_correctoin (m1, m2, distance, velocity, version):
    """
    Calculates GR correctoin
    Args:
        m1: mass of first body
        m2: mass of second body
        distance: three dimensional distance array
        velocity: three dimensional velocity array

    Returns: A numpy array with the three force correction components

    """
    dist_norm = np.sum(distance ** 2.) ** 0.5
    velocity_norm = np.sum(velocity ** 2) ** 0.5
    beta = velocity_norm / c # total beta factor
    L = np.sum((np.cross(velocity, distance))**2)**0.5 #total angular momentum per unit mass normalised
    beta_version = (G * m1 * m2 * distance / dist_norm ** 3.) * (1 + (3 * beta ** 2))
    angular_version = (G * m1 * m2 * distance / dist_norm ** 3.) * (1 + (3 * L**2)/(c**2 * dist_norm ** 2))
    if version == 'b' :
        return beta_version
    elif version == 'L':
        return angular_version
    else:
        return 'Version not defined'


def force_newton(m1, m2, distance):
    """
    Calculate the force using Newton's law
    :param m1: mass of first body
    :param m2: mass of second body
    :param distance:
    :return:
    A numpy array with the three force components
    """
    # Calculate the acceleration using Newtonian Gravity
    dist_norm = np.sum(distance ** 2.) ** 0.5
    return G * m1 * m2 * distance / dist_norm ** 3.


# Sun
mass_sun = 1
pos_sun = np.array([7.93567917e-03, -6.29360340e-04, -2.31793679e-04])
vel_sun = np.array([3.56426004e-06, 7.70848450e-06, -1.38462510e-07])


# Mercury
mass_mercury = 0.33011 * 10 ** 24 / MSUN
pos_mercury = np.array([-5.78670715e-02, -4.61182491e-01, -3.17988125e-02])  # AU
vel_mercury = np.array([2.22124712e-02, -2.53545004e-03, -2.24740703e-03])  # AU/DAY

distance = pos_mercury - pos_sun
velocity = vel_mercury - vel_sun

GR_beta = (np.sum((GR_correctoin (mass_mercury, mass_sun, distance, velocity, version = 'b'))**2))**0.5
GR_L = (np.sum((GR_correctoin (mass_mercury, mass_sun, distance, velocity, version = 'L'))**2))**0.5
N = (np.sum((force_newton (mass_mercury, mass_sun, distance))**2))**0.5

difference_beta = GR_beta - N
difference_L = GR_L -N
'''
print(f'Newtonian Force= {N}')
print(f'GR Force Beta version= {GR_beta}')
print(f'GR Force L version= {GR_L}')
print(f'Difference (beta version)= {difference_beta}')
print(f'Difference (L version)= {difference_L}')
print(f'Order= {GR_L/N}')
'''

N_file = open('newton_simulation.pickle', 'rb')
N_system = pickle.load(N_file)
GR_file = open('GR_simulation.pickle', 'rb')
GR_system = pickle.load(GR_file)


x_GR = GR_system.get_positions()
x_N = N_system.get_positions()

GR_sun = x_GR[:, 0, :] - x_GR[:, 0, :]
GR_mercury = x_GR[:, 1, :] - x_GR[:, 0, :]

x_GR_2 = np.concatenate([GR_sun, GR_mercury], axis=-1)

print(x_GR[:1, :, :])

print(x_GR_2[:1, :, :])
