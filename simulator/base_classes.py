import numpy as np


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

