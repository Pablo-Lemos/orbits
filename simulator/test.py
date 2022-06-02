import numpy as np


pos_mercury = np.array([-5.78670715e-02, -4.61182491e-01, -3.17988125e-02])  # AU
vel_mercury = np.array([2.22124712e-02, -2.53545004e-03, -2.24740703e-03])  # AU/DAY

L = np.sum(np.cross(vel_mercury, pos_mercury)**2)**0.5

print(L**2)