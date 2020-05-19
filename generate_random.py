from simulate_orbits import *
import numpy as np

def random(shape, minval, maxval):
    return (maxval - minval) * np.random.rand(shape) + minval

delta_time = 0.1*DAY
total_time = 10000*DAY
nplanets = 1

noise = random(nplanets, -0.1, 0.1)

def get_orbital_velocity(radius, mass_planet, mass_star, G, noise = 0): 
    """Calculate the velocity required to keep an orbit

    Parameters
    ---------- 
    radius : float
      the radial distance from the body to the center of the orbit
    mass_planet : float
      mass of the planet orbiting
    mass_star : float
      mass of the body at the center of the orbit
    G : float 
      Gravitational Constant
    noise: float
      The maximum fractional amount by which the initial velocity is perturbed. 
      Defaults to zero

    Returns
    ---------- 
    velocity : float
      the orbital velocity

    """

    velocity = np.sqrt(G*mass_star/radius)

    # Add noise
    if noise > 0:
        delta_vel = rand.uniform(*(-noise,noise), size = np.shape(velocity))
        velocity *= (1.+delta_vel)

    return velocity

# Mercury with random initial conditions
for i in range(nplanets):
    # Sun
    sun = Body()
    sun.name = 'star_'+str(i)
    sun.mass = 1.98892 * 10**30 # kg
    sun.pos = np.zeros(3) # m

    mercury = Body()
    mercury.name = 'planet_'+str(i)
    mercury.mass = 0.33011 * 10**24 #kg
    pos_mercury = 0.387*AU
    vel_mercury = get_orbital_velocity(pos_mercury, mercury.mass, sun.mass, G)
    mercury.initiate(pos_mercury, vel_mercury)

    #sun.vel = np.zeros(3) 
    sun.vel = -np.array(mercury.vel)*mercury.mass/sun.mass

    planets = [sun, mercury]
    simulate(planets, total_time, delta_time)
