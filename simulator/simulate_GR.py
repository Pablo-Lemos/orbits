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
AU = 149.6e9     # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY
MSUN = 1.98892 * 10**30 # Solar mass
MEARTH = 5.9742 * 10**24 # Earth mass
G = 6.67428e-11/AU**3*MSUN*DAY**2 # The gravitational constant G in AU**3 /MSUN/ Day^2


def simulate(bodies, total_time, delta_time, force_law):
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

def example(force_law):
    """
    The main function. Defines the bodies and parameters to be used in the simulation,
    and starts it
    """

    delta_time = 2*(24/24.) # The time interval to be used in Days
    total_time = 1000*88. # Total time of the Simulation in Days

    # Define Astronomical bodies. Data taken from:
    # http://nssdc.gsfc.nasa.gov/planetary/factsheet/

    # Sun
    sun = Body(name='Sun', mass=1.)
    # Start the Sun at the origin with no velocity
    pos_sun = np.array([7.93567917e-03, -6.29360340e-04, -2.31793679e-04])
    vel_sun = np.array([3.56426004e-06, 7.70848450e-06, -1.38462510e-07])
    sun.initiate(pos=pos_sun, vel=vel_sun)

    # Mercury
    mercury = Body(name='Mercury', mass=0.33011 * 10 ** 24 / MSUN)
    pos_mercury = np.array([-5.78670715e-02, -4.61182491e-01, -3.17988125e-02])  # AU
    vel_mercury = np.array([2.22124712e-02, -2.53545004e-03, -2.24740703e-03])  # AU/DAY
    mercury.initiate(pos_mercury, vel_mercury)

    # Venus
    venus = Body(name='Venus', mass=4.8685 * 10 ** 24 / MSUN)
    pos_venus = np.array([7.25372142e-01, 1.02962658e-01, -4.02455202e-02])  # AU
    vel_venus = np.array([-2.96452677e-03, 1.99351788e-02, 4.42465220e-04])  # AU/DAY
    venus.initiate(pos_venus, vel_venus)

    # Earth
    earth = Body(name='Earth', mass=MEARTH / MSUN)
    pos_earth = np.array([-2.81758546e-01, 9.39043493e-01, -1.91271807e-04])  # AU
    vel_earth = np.array([-1.67163499e-02, -5.11906912e-03, -1.03151390e-06])  # AU/DAY
    earth.initiate(pos_earth, vel_earth)

    '''
    # Jupiter
    jupiter = Body(name='Jupiter', mass=1.898 * 10 ** 27 / MSUN)
    pos_jupiter = 5.203736631  # AU
    vel_jupiter = 7.50802139e-03  # AU/DAY
    jupiter.initiate(pos_jupiter, vel_jupiter)
    '''

    #Run the simulation
    simulate([sun, mercury, venus, earth], total_time, delta_time, force_law=force_law)

    return StarSystem([sun, mercury, venus, earth])

if __name__ == '__main__':
    #GR
    print("Running GR Simulation...")
    simulation_GR = example(force_law='GR')
    print("GR Simulation is complete")
    # To save this:
    file = open('GR_simulation.pickle', 'wb')
    pickle.dump(simulation_GR, file)
    print("GR Simulation saved")

    #N
    print("Running Newtonian Simulation...")
    simulation_N = example(force_law='N')
    print("Newtonian Simulation is complete")
    # To save this:
    file = open('Newton_simulation.pickle', 'wb')
    pickle.dump(simulation_N, file)
    print("Newtonian Simulation saved")
