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
        orbit = np.loadtxt(path+name+'.txt', usecols = [2,3,4,5,6,7], 
                            unpack=True, delimiter=',')
    except IndexError:
        orbit = np.genfromtxt(path+name+'.txt', usecols = [2,3,4,5,6,7], 
                            unpack=True, delimiter=',',
                            skip_header=22, skip_footer = 31
                            )

    return orbit

def add_moons(name, orbits_ls, masses_ls, names_ls, use_moons, path):
    ''' If not using moons, corrects the orbit by changing it to the center 
    of mass of the planet + moons system. Else, add the moons to the list of 
    orbits
    Parameters: 
    -----------
    name: str
        the name of the planet
    orbits_ls : ls
        a list containing the orbits data
    masses_ls : ls
        the masses of all bodies
    names_ls : ls
        the names of all bodies
    use_moons: bool
        whether to treat the moons as separate bodies.
    path: string
        the path to the orbit files

    Returns: 
    --------
    orbits_ls : ls
        a list containing the updated orbits data 
    masses_ls : ls
        the masses of all bodies
    names_ls : ls
        the names of all bodies
    '''   
    # Check if the planets has moons
    if name in planets_with_moons:
        i = planet_names.index(name)
        j = planets_with_moons.index(name)
        for (moon, mass_moon) in zip(moon_names[j],moon_masses[j]):
            orbit_moon = read_orbit(moon, path)
            # If the moons are not being treated as separate bodies, we change
            # the orbit of the planets by the orbit of the Center of Mass of 
            # the planet and orbit system. 
            if use_moons == False:
                #print('Adding data for', moon)
                orbits_ls[-1] += orbit_moon*mass_moon/planet_masses[i]

            # Else, add each moon
            else:
                print('Reading data for', moon)
                orbits_ls.append(orbit_moon)
                masses_ls.append(mass_moon)
                names_ls.append(moon)

    
    return orbits_ls, masses_ls, names_ls


def main(nplanets = 0, 
              use_moons = False, 
              frame = 'b', 
              path = './nasa_orbits/'):
    ''' Reads the data files and returns a numpy array with the orbits 
    Parameters: 
    -----------
    nplanets: int
        the number of planets to be used. If 0, use all the planets. 
        Defaults to 0
    use_moons: bool
        whether to treat the moons as separate bodies. Defaults to False
    frame: str
        The frame of reference to be used. Options are 'b' (barycenter) or 
        's' (sun_center). Defaults to 'barycenter'
    path: string
        the path to the orbit files. It should contain two folders: 'barycenter'
        and 'sun_center'

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

    # Modify the path to include the reference frame
    if frame[0]=='b':
        print('Reading data in Solar System barycenter reference frame')
        path += 'barycenter/'
    elif frame[0]=='s':
        print('Reading data in Sun reference frame. Known masses will then be', \
            'used to move to barycenter frame of used bodies. Use this only for', \
            'testing and when few bodies are being used.')
        path += 'sun_center/'    
    else: 
        raise ValueError("frame must be 'b' (barycenter) or 's' (sun_center).")

    # If using the default value for nplanets, we use all 8
    if nplanets == 0: 
        nplanets = 8

    # Read the sun's orbit
    orbit_sun = read_orbit('sun', path)
    # Create a list that will contain all the orbits
    orbits_ls = [orbit_sun]
    # Create a list that will contain all the masses
    masses_ls = [sun_mass]
    names_ls = ['sun']

    for i in range(nplanets):
        name = planet_names[i]
        print('Reading data for', name)
        orbit = read_orbit(name, path)
        orbits_ls.append(orbit)
        masses_ls.append(planet_masses[i])
        names_ls.append(name)
        orbits_ls, masses_ls, names_ls = add_moons(name, orbits_ls, masses_ls,
                                         names_ls, use_moons, path)

    orbits_data = np.stack(orbits_ls)
    # Transpose to get an array with time, planet, axes
    orbits_data = orbits_data.transpose(2,0,1)
    
    # Convert list of masses to numpy array
    masses = np.asarray(masses_ls)

    print('Finished reading data')
    print('The data array contains', len(orbits_data[0]), 'bodies.')

    # If the frame is sub center, need to change things.
    if frame[0] == 's':
        print("Changing frame of reference (this only happens when the data", \
           "is loaded in the Sun's frame). ")
        # Change frame of reference to the barycenter of the planets we are using
        P = masses[np.newaxis, :, np.newaxis]*orbits_data[:,:,3:] 
        V_ref = np.sum(P, axis = 1,keepdims=True)/np.sum(masses)
        orbits_data[:,:,3:] -= V_ref

    return orbits_data, masses, names_ls


                    
