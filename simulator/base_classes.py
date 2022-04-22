import types

import numpy as np

#Define constants
AU = 149.6e6 * 1000     # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY
MSUN = 1.98892 * 10**30 # Solar mass
MEARTH = 5.9742 * 10**24 # Earth mass
G = 6.67428e-11/AU**3*MSUN*DAY**2 # The gravitational constant G in AU**3 /MSUN/ YEAR^2

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


class Body(object):
    def __init__(self, mass=0., name=''):
        self._mass = mass
        self._name = name
        self._positions = None
        self._velocities = None
        self._currPos = None
        self._currVel = None
        self._currAcc = None

    def get_mass(self):
        return self._mass

    def get_name(self):
        return self._name

    def get_positions(self):
        return self._positions

    def get_velocities(self):
        return self._velocities

    def get_current_position(self):
        return self._currPos

    def get_current_velocity(self):
        return self._currVel

    def add_trajectory(self, data):
        assert (len(data.shape) == 2), "Data must be 2D (time, X)"
        assert (data.shape[1] == 6), "Wrong data dimensions"

        self._positions = data[:, :3]
        self._velocities = data[:, 3:]

    def initiate(self, pos, vel):
        """
        Initiate the position of the body, from either 3D arrays, or scalars
        containing the magnitudes, in which case the direction is chosen
        randomly
        """
        if isinstance(pos, (int, float)):
            # Create unitary vector, for now, we keep z = 0
            u = random_two_vector()

            # Define the initial position
            x, y = pos * u
            self._currPos = np.array([x, y, 0])

        elif len(pos) == 3:
            pos = np.array(pos)
            u = pos / np.linalg.norm(pos)
            self._currPos = pos

        else:
            raise "Wrong format for position, must be scalar or 3d array"

        if isinstance(vel, (int, float)):
            # Define the initial velocity
            vy, vx = vel * u
            self._currVel = np.array([vx, -vy, 0])

        elif len(vel) == 3:
            self._currVel = vel

        else:
            raise "Wrong format for velocity, must be scalar or 3d array"

        if self._positions is None:
            self._positions = self._currPos.reshape(1, 3)
        if self._velocities is None:
            self._velocities = self._currVel.reshape(1, 3)

    def interaction(self, other, force_law=None):
        """
        Returns the acceleration due to gravitational interaction with
        another body

        Parameters
        ----------
        other : Body
            The astronomical body whose gravitational pull we are computing
        """

        # Compute distance to the other body
        distance = other.get_current_position() - self._currPos

        if force_law is None:
            force_law = force_newton
        assert hasattr(force_law, '__call__')
        assert force_law.__code__.co_argcount == 3, "Force law must have 3 " \
                                                    "arguments: m1, m2, x"

        self._currAcc += force_law(self._mass, other.get_mass(), distance) / self._mass

    def update(self, delta_time):
        """Updates the position and velocity of the body after a time step

        Parameters
        ----------
        delta_time : float
            The size of the time step in seconds
        """
        self._currVel += self._currAcc * delta_time
        self._currPos += self._currVel * delta_time

        self._positions = np.concatenate([self._positions,
                                          self._currPos.reshape(1, 3)], axis=0)
        self._velocities = np.concatenate([self._velocities,
                                          self._currVel.reshape(1, 3)], axis=0)

    def reset_acceleration(self):
        self._currAcc = np.zeros(3)


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
            self._names.append(body.get_name())
        return self._names

    def get_masses(self):
        if len(self._masses) > 0:
            return self._masses

        for body in self._bodies:
            self._masses.append(body.get_mass())
        self._masses = np.array(self._masses)
        return self._masses

    def get_positions(self):
        orbits = []
        timeSteps = np.infty
        for body in self._bodies:
            x = body.get_positions()
            currSteps = x.shape[0]
            if currSteps < timeSteps:
                timeSteps = currSteps
            orbits.append(x[:timeSteps])

        orbits = np.stack(orbits)
        # Transpose to get an array with time, planet, axes
        return orbits.transpose(1, 0, 2)

    def get_velocities(self):
        orbits = []
        timeSteps = np.infty
        for body in self._bodies:
            v = body.get_velocities()
            currSteps = v.shape[0]
            if currSteps < timeSteps:
                timeSteps = currSteps
            orbits.append(v[:timeSteps])

        orbits = np.stack(orbits)
        # Transpose to get an array with time, planet, axes
        return orbits.transpose(1, 0, 2)

