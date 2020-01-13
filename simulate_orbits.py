#!/usr/bin/env python3

'''
Code that simulates the orbits of solar system bodies assuming Newtonian 
gravity. 

Created by Pablo Lemos (UCL) 
28-11-2019
pablo.lemos.18@ucl.ac.uk
'''

import numpy as np
import pandas as pd

#Define constants
G = 6.67428e-11 # The gravitational constant G in N m^2 / kg^2
AU = 149.6e6 * 1000     # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds

class Body:
    """
    A class to represent astronomical bodies (e.g. planets and stars)

    Attributes
    ---------- 
    name : str
        the name of the body
    mass : float
        mass in kg
    pos : float(3) 
        position of the body as array (x,y,z) in m
    vel : float(3) 
        velcity of the body as array (vx, vy, vz) in m/s

    Methods
    -------
    interaction(other) 
        Returns the acceleration due to gravitational interaction with another
        body
    update(delta_time)
        Updates the position and velocity of the body after a time step
    """
    
    #def __init__(self, name, mass, pos, vel, acc): 
    def __init__(self):
        """
        Parameters
        ---------- 
        name : str
            the name of the body
        mass : float
            mass in kg
        pos : float(3) 
            position of the body as array (x,y,z) in m
        vel : float(3) 
            velocity of the body as array (vx, vy, vz) in m/s
        acc : float(3) 
            acceleration of the body as array (ax, ay, az) in m/s^2
        orbit : ls
            a list where the orbit is stored
        """

        self.name = "" 
        self.mass = 0.
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.orbit = []

    def interaction(self, other):
        """Returns the acceleration due to gravitational interaction with 
        another body
        
        Parameters 
        ----------
        other : Body
            The astronomical body whose gravitational pull we are computing
        """
    
        # Compute distance to the other body
        delta_pos = other.pos - self.pos
        dist = np.sum(delta_pos**2.)**0.5
        
        #Calculate the acceleration using Newtonian Gravity
        self.acc += G*other.mass*delta_pos/dist**3.

    def update(self, delta_time):
        """Updates the position and velocity of the body after a time step

        Parameters
        ----------
        delta_t : float 
            The size of the time step in seconds
        """

        self.vel += self.acc*delta_time
        self.pos += self.vel*delta_time

def simulate(bodies, total_time, delta_time):
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

    time = 0 # Current time

    while time<total_time: 
        for body in bodies: 
            body.acc = (0,0,0) # Restart acceleration
            for other_body in bodies: 
                if body is not other_body: # Not sum over interaction with self
                    # Sum over interactions with all other bodies
                    body.interaction(other_body) 

            body.update(delta_time) # Update position and velocity of each body
            body.orbit.append(body.pos/AU) # Store position of each body (in AU)

        time += delta_time # Update total time

    # Create the pandas DataFrame for each orbit
    for body in bodies:
        file_name = './orbits/'+body.name+'.dat' # File name to use for saving
        
        # Store as pandas array
        # df = pd.DataFrame(body.orbit, columns = ['x[AU]', 'y[AU]', 'z[AU]'])
        # df.to_pickle(file_name)
 
        # Store as numpy array
        ar = np.asarray(body.orbit) #convert orbit into numpy array
        np.savetxt(file_name, ar, header = 'px[AU]    py[AU]    vx[m/s]    vy[m/s]') 

def main():
    """
    The main function. Defines the bodies and parameters to be used in the simulation, 
    and starts it
    """

    delta_time = 1*DAY # The time interval to be used
    total_time = 400*DAY # Total time of the Simulation

    # Define Astronomical bodies. Data taken from: 
    # http://nssdc.gsfc.nasa.gov/planetary/factsheet/

    # Sun
    sun = Body() 
    sun.name = 'Sun'
    sun.mass = 1.98892 * 10**30 # kg
    sun.pos = np.zeros(3) # m
    sun.vel = np.zeros(3) # m/s

    # Earth
    earth = Body()
    earth.name = 'Earth'
    earth.mass = 5.9742 * 10**24 # kg
    earth.pos = np.array([-1*AU,0.,0.]) # m
    earth.vel = np.array([0.,29.783*1000,0.])# m/sec

    #Run the simulation
    simulate([sun, earth], total_time, delta_time)

if __name__ == '__main__':
    main()
