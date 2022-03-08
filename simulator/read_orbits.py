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

from data.solar_system_names import *
from simulator.base_classes import *
import os
import pickle

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
        orbit = np.loadtxt(os.path.join(path, name + '.txt'), usecols=[2, 3, 4, 5, 6, 7],
                           unpack=True, delimiter=',')
    except IndexError:
        orbit = np.genfromtxt(os.path.join(path, name + '.txt'), usecols=[2, 3, 4, 5, 6, 7],
                              unpack=True, delimiter=',',
                              skip_header=22, skip_footer=31
                              )

    return orbit.T


def main(nplanets=0,
         path=None,
         read_data=True):
    ''' Reads the data files and returns a numpy array with the orbits 
    Parameters: 
    -----------
    nplanets: int
        the number of planets to be used. If 0, use all the planets. 
        Defaults to 0
    path: string
        the path to the orbit files. It should contain a folder: 'barycenter'
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

    if not path:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, '../data/')

    print('Reading data in Solar System barycenter reference frame')
    path = os.path.join(path, 'barycenter')

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
            j = planets_with_moons.index(name)
            for (name_moon, mass_moon) in zip(moon_names[j], moon_masses[j]):
                moon = Body(mass=mass_moon, name=name_moon)
                if read_data:
                    print('Reading data for', name_moon)
                    orbit = read_orbit(name_moon, path)
                    moon.add_trajectory(orbit)

                bodies.append(moon)

    return StarSystem(bodies)


if __name__ == "__main__":
    system = main(nplanets=1)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, '../data/debug.pkl')
    file = open(path, 'wb')
    pickle.dump(system, file)
