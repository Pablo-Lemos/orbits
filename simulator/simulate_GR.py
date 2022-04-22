#!/usr/bin/env python3

'''
Code that simulates the orbits of solar system bodies for a given force law
(Will be used for force law = Newtons Law + GR correction)

Created by Pablo Lemos (UCL)
28-11-2019
pablo.lemos.18@ucl.ac.uk
'''

import numpy as np
from base_classes_GR import *
import pickle

#Define constants
AU = 149.6e6 * 1000     # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY
MSUN = 1.98892 * 10**30 # Solar mass
MEARTH = 5.9742 * 10**24 # Earth mass
G = 6.67428e-11/AU**3*MSUN*DAY**2 # The gravitational constant G in AU**3 /MSUN/ YEAR^2


def simulate_GR(bodies, total_time, delta_time, force_law=None):
    """
    Simulates the orbits for a certain period of time, and stores the results
    as a panda array.

    Parameters
    ----------
    bodies : list
        The bodies that will interact in the simulation
    total_time : float
        The amount of time (in seconds) that the simulation will last for
    time_step : float
        The size of the time steps in the simulation in seconds
    """

    time = 0  # Current time
    numSteps = int(total_time // delta_time)
    numBodies = int(len(bodies))
    orbits = np.empty([numSteps, numBodies, 6])

    for i in range(numSteps):
        j = 0
        for body in bodies:
            body.reset_acceleration()  # Restart acceleration
            for other_body in bodies:
                if body is not other_body:  # Not sum over interaction with self
                    # Sum over interactions with all other bodies
                    body.interaction(other_body, force_law)

            body.update(delta_time)  # Update position and velocity of each body
            # orbits[i, j, :] = np.concatenate(
            #     [body.get_current_position(), body.get_current_velocity()])  # Store position of
            # # each body (in AU)
            j += 1

        time += delta_time  # Update total time

def example():
    """
    The main function. Defines the bodies and parameters to be used in the simulation,
    and starts it
    """

    delta_time = (1/24.)*DAY/YEAR # The time interval to be used in years (1 hour)
    total_time = 1. # Total time of the Simulation in years

    # Define Astronomical bodies. Data taken from:
    # http://nssdc.gsfc.nasa.gov/planetary/factsheet/

    # Sun
    sun = Body(name = 'Sun', mass = 1.)
    # Start the Sun at the origin with no velocity
    sun.initiate(pos = np.zeros(3), vel = np.zeros(3))

    # Mercury
    mercury = Body(name = 'Mercury', mass = 0.33011 * 10**24/MSUN)
    pos_mercury = np.array([0.387, 0., 0.]) #AU
    vel_mercury = np.array([0., -47.36 * 1000/AU*DAY, 0.]) #AU/YEAR
    mercury.initiate(pos_mercury, vel_mercury)

    #Venus
    #venus = Body(name = 'Venus', mass = 4.8685 * 10**24/MSUN)
    #pos_venus = np.array([0.723, 0., 0.]) #AU
    #vel_venus = np.array([0.,-35.02 * 1000/AU*DAY, 0.]) #AU/Y
    #venus.initiate(pos_venus, vel_venus)

    # Earth
    #earth = Body(name = 'Earth', mass = MEARTH/MSUN)
    #pos_earth = np.array([-1.,0.,0.]) # AU
    #vel_earth = np.array([0.,29.783*1000/AU*DAY,0.])# AU/Y
    #earth.initiate(pos_earth, vel_earth)

    #Run the simulation
    simulate_GR([sun, mercury], total_time, delta_time)

    return StarSystem([sun, mercury])

if __name__ == '__main__':
    simulation_GR = example()
    # To save this:
    # import pickle
    file = open('GR_simulation.pickle', 'wb')
    pickle.dump(simulation_GR, file)
