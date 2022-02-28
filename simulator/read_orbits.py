'''
Read NASA planetary orbits, download from https://ssd.jpl.nasa.gov/horizons.cgi#top
with format X, Y, Z, VX, VY, VZ format them, and return two data vectors, one 
containing positions and one containing accelerations. 

For the future: Create a class that contains orbits, names and masses. Will
improve appearance.

Written by Pablo Lemos (UCL)
pablo.lemos.18@ucl.ac.uk
Nov 2020
'''

import numpy as np
from solar_system_names import *


class Body(object):
    def __init__(self, mass, name):
        self._mass = mass
        self._name = name
        self._positions = None
        self._velocities = None

    def get_mass(self):
        return self._mass

    def get_name(self):
        return self._name

    def get_positions(self):
        return self._positions

    def get_velocities(self):
        return self._velocities

    def add_trajectory(self, data):
        assert (len(data.shape) == 2), "Data must be 2D (time, X)"
        assert (data.shape[1] == 6), "Wrong data dimensions"

        self._positions = data[:, :3]
        self._velocities = data[:, 3:]


class StarSystem(object):
    def __init__(self, bodies):
        self._bodies = bodies
        self._names = []
        self._masses = []
        self.numPlanets = len(bodies)
        self.numEdges = self.numPlanets * (self.numPlanets - 1) // 2
        self._positions = None
        self._velocities = None

    def get_names(self):
        if len(self._names) > 0:
            return self._names

        for body in self._bodies:
            self._names.append(body.name)
        return self._names

    def get_masses(self):
        if len(self._masses) > 0:
            return self._masses

        for body in self._bodies:
            self._masses.append(body.mass)
        self._masses = np.array(self._masses)
        return self._masses

    def get_positions(self):
        orbits = []
        for body in self._bodies:
            orbits.append(body.get_positions())

        orbits = np.stack(orbits)
        # Transpose to get an array with time, planet, axes
        return orbits.transpose(1, 0, 2)

    def get_velocities(self):
        orbits = []
        for body in self._bodies:
            orbits.append(body.get_velocities())

        orbits = np.stack(orbits)
        # Transpose to get an array with time, planet, axes
        return orbits.transpose(1, 0, 2)


def read_orbit(name, path):
    ''' Reads the data for a single orbit
    Parameters: 
    -----------
    name: str
        the name of the body
    path: string
        the path to the orbit file

    Returns: 
    --------
    orbit_data : np.array
        a numpy array containing the positions and accelerations for the body 
    '''
    try:
        orbit = np.loadtxt(path + name + '.txt', usecols=[2, 3, 4, 5, 6, 7],
                           unpack=True, delimiter=',')
    except IndexError:
        orbit = np.genfromtxt(path + name + '.txt', usecols=[2, 3, 4, 5, 6, 7],
                              unpack=True, delimiter=',',
                              skip_header=22, skip_footer=31
                              )

    return orbit.T


def main(nplanets=0,
         use_moons=True,
         path='./nasa_data/',
         read_data=True):
    ''' Reads the data files and returns a numpy array with the orbits 
    Parameters: 
    -----------
    nplanets: int
        the number of planets to be used. If 0, use all the planets. 
        Defaults to 0
    use_moons: bool
        whether to treat the moons as separate bodies. Defaults to False
    path: string
        the path to the orbit files. It should contain two folders: 'barycenter'
        and 'sun_center'
    read_data: bool
        whether to read data files, or just masses and names, Defauls to true

    Returns: 
    --------
    orbits_data : np.array()
        a numpy array containing the positions and accelerations for each 
        body.
    masses : np.array()
        the masses of all bodies
    names_ls : ls
        the names of all bodies
    '''

    print('Reading data in Solar System barycenter reference frame')
    path += 'barycenter/'

    # If using the default value for nplanets, we use all 8
    if nplanets == 0:
        nplanets = 8

    sun = Body(name='sun', mass=sun_mass)

    if read_data:
        # Read the sun's orbit
        orbit_sun = read_orbit('sun', path)
        # Create a list that will contain all the orbits
        sun.add_trajectory(orbit_sun)

    # Create a list that will contain all the bodies
    bodies = [sun]

    for i in range(nplanets):
        name = planet_names[i]
        planet = Body(mass=planet_masses[i], name=name)
        if read_data:
            print('Reading data for', name)
            orbit = read_orbit(name, path)
            planet.add_trajectory(orbit)

        bodies.append(planet)

        if name in planets_with_moons:
            for (name_moon, mass_moon) in zip(moon_names[j], moon_masses[j]):
                moon = Body(mass=mass_moon, name=name_moon)
                if read_data:
                    print('Reading data for', name)
                    orbit = read_orbit(name_moon, path)
                    moon.add_trajectory(orbit)

            bodies.append(moon)

    return StarSystem(bodies)


if __name__ == "__main__":
    system = main(nplanets=1)
    X = system.get_positions()
    print(X.shape)
