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
AU = 149.6e6 * 1000     # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY
MSUN = 1.98892 * 10**30 # Solar mass
MEARTH = 5.9742 * 10**24 # Earth mass
G = 6.67428e-11/AU**3*MSUN*DAY**2 # The gravitational constant G in AU**3 /MSUN/ YEAR^2

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
    acc : float(3) 
        acceleration of the body as array (ax, ay, az) in m/s^2
    orbit : ls
        a list where the orbit is stored

    Methods
    -------
    interaction(other) 
        Returns the acceleration due to gravitational interaction with another
        body
    update(delta_time)
        Updates the position and velocity of the body after a time step
    """
    
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
    def initiate(self, radial_pos, total_vel):
        """
        Randomly generate an initial position and velocity given a radial position
        and a total velocity. 

        """

        # Create unitary vector, for now, we keep z = 0
        u = random_two_vector()

        # Define the initial position
        x, y = radial_pos*u
        z = 0.

        # Define the initial velocity 
        vy, vx = total_vel*u

        self.pos = np.array([x, y, 0])
        self.vel = np.array([vx, -vy, 0])

    def interaction(self, other, G):
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

def random_two_vector():
    """
    Generates a random 2D unitary vector

    Returns:
    --------
    x,y: float
        Coordinates of the unitary vector (x^2 + y^2 = 1)

    """
    phi = np.random.uniform(0,np.pi*2)
    x = np.cos(phi)
    y = np.sin(phi)
    return np.array([x,y])

def simulate(bodies, total_time, delta_time, G, save = False):
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
    nsteps = int(total_time//delta_time)
    nbodies = int(len(bodies))
    orbits = np.empty([nsteps, nbodies, 6])

    for i in range(nsteps):
        j=0
        for body in bodies: 
            body.acc = (0,0,0) # Restart acceleration
            for other_body in bodies: 
                if body is not other_body: # Not sum over interaction with self
                    # Sum over interactions with all other bodies
                    body.interaction(other_body, G) 

            body.update(delta_time) # Update position and velocity of each body
            orbits[i, j, :] = np.concatenate([body.pos, body.vel]) # Store position of each body (in AU)
            j+=1 

        time += delta_time # Update total time

    if save == True:
        # Create the pandas DataFrame for each orbit
        orbits = []
        for body in bodies:
            file_name = './orbits/'+body.name # File name to use for saving
            
            # Store as pandas array
            # df = pd.DataFrame(body.orbit, columns = ['x[AU]', 'y[AU]', 'z[AU]'])
            # df.to_pickle(file_name)
    
            # Store as numpy array
            orbits.append(np.asarray(body.orbit)) #convert orbit into numpy array

        np.save(file_name, orbits)#, header = 'x[AU]', 'y[AU]', 'z[AU]') 

    return orbits

def main():
    """
    The main function. Defines the bodies and parameters to be used in the simulation, 
    and starts it
    """

    delta_time = (1/24.)*DAY/YEAR # The time interval to be used in years (1 hour)
    total_time = 1. # Total time of the Simulation in years

    # Define Astronomical bodies. Data taken from: 
    # http://nssdc.gsfc.nasa.gov/planetary/factsheet/

    # Sun
    sun = Body() 
    sun.name = 'Sun'
    sun.mass = 1. # Solar masses
    sun.pos = np.zeros(3)  
    sun.vel = np.zeros(3) 

    # Mercury
    mercury = Body()
    mercury.name = 'Mercury'
    mercury.mass = 0.33011 * 10**24/MSUN # Solar masses
    mercury.pos = np.array([0.387, 0., 0.]) #AU
    mercury.vel = np.array([0., -47.36 * 1000/AU*DAY, 0.]) #AU/YEAR 

    #Venus
    venus = Body()
    venus.name = 'Venus'
    venus.mass = 4.8685 * 10**24/MSUN # Solar masses
    venus.pos = np.array([0.723, 0., 0.]) #AU
    venus.vel = np.array([0.,-35.02 * 1000/AU*DAY, 0.]) #AU/Y

    # Earth
    earth = Body()
    earth.name = 'Earth'
    earth.mass = MEARTH/MSUN
    earth.pos = np.array([-1.,0.,0.]) # AU
    earth.vel = np.array([0.,29.783*1000/AU*DAY,0.])# AU/Y

    #Run the simulation
    orbits = simulate([sun, mercury, venus, earth], total_time, delta_time, G)

    return orbits

if __name__ == '__main__':
    main()
