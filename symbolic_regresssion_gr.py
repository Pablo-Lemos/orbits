from planets_tf2_gr import read_data
from ml_model_gr import LearnForces, Normalize_gn
import helper_functions_gr as hf
import tensorflow as tf
import numpy as np
from pysr import PySRRegressor
from pysr import best
import os
import sympy

# Training variables
num_time_steps_tr = 504000  # Number of time steps for training 30y=504000; 40y=680800; 50y=856000; 60y=1031200
noise_level = 0.01  # Standard deviation of Gaussian noise for randomly perturbing input data
# One time step is 30 minutes
# An orbit for saturn is 129110 steps
num_time_steps_val = 10000  # Using few to speed up calculations

# Global constants
AU = 149.6e6 * 1000  # Astronomical Unit in meters.
DAY = 24 * 3600.  # Day in seconds
YEAR = 365.25 * DAY  # Year
delta_time = (0.5 / 24.)  # 30 minutes
MSUN = 1.9885e+30  # kg
MEARTH = 5.9724e+24  # kg
G = 6.67428e-11 / AU ** 3 * MSUN * DAY ** 2  # Change units of G to AU^3 MSun^{-1} Day^{-2}
A_norm = 0.00042411583592113497  # From planets_tf2 (I will change the way this is stored eventually)
c = (2.99792458 * 10**8) * DAY / AU # speed of light in AU/Day


def force_newton(m1, m2, x):
    return G * m1 * m2 / np.linalg.norm(x) ** 3. * x


def GR_correctoin (m1, m2, distance, velocity):
    """
    Calculates GR correction
    Args:
        m1: mass of first body
        m2: mass of second body
        distance: three dimensional distance array
        velocity: three dimensional velocity array

    Returns: A numpy array with the three force correction components

    """
    dist_norm = np.sum(distance ** 2.) ** 0.5
    L_norm = np.linalg.norm(np.cross(distance, velocity))
    f_n = G * m1 * m2 * distance / dist_norm ** 3.
    corr = 3.*L_norm**2/(c**2 * dist_norm**2)
    return f_n * (1 + 1000 * corr)


def load_model(system, norm_layer, senders, receivers):
    """ Load the model"""

    # Restore best weights not working, but found way around using checkpoint
    checkpoint_filepath = './saved_models/sun_mercury_1000_gr_60y_1'

    # Create a model
    model = LearnForces(system.numPlanets, senders, receivers, norm_layer,
                        noise_level=noise_level)

    # Compile
    model.compile()

    model.load_weights(checkpoint_filepath)

    return model


def format_data_symreg(data_tr, data_symreg, system):
    """
    Convert the data into normalized tensorflow data objects that we can
    use for training
    """
    #print(f'Data_tr shape: {data_tr.shape}')
    #print(f'Data_symreg shape: {data_symreg.shape}')

    nedges = system.numEdges
    masses = system.get_masses()

    # Create empty arrays for the distances for training and validation
    D_V_tr = np.empty([len(data_tr), nedges, 6])
    D_V_symreg = np.empty([len(data_symreg), nedges, 6])
    F_symreg = np.empty([len(data_symreg), nedges, 3])
    Fn_symreg = np.empty([len(data_symreg), nedges, 3])

    k = 0
    # Create empty lists for the senders and receivers that will be used for
    # the edges of the graph
    senders, receivers = [], []
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                # For every pair of objects, assign a distance, a velocity, a GR Force, and a Newtonian Force
                D_V_tr[:, k, :] = data_tr[:, j, :6] - data_tr[:, i, :6]
                D_V_symreg[:, k, :] = data_symreg[:, j, :6] - \
                                    data_symreg[:, i, :6]
                F_symreg[:, k, :] = GR_correctoin(masses[i], masses[j],
                    D_V_symreg[:, k, :3], D_V_symreg[:, k, 3:6])
                Fn_symreg[:, k, :] = force_newton(masses[i], masses[j], D_V_symreg[:, k, :3])

                k += 1
                # Add sender and receiver index
                receivers.append(i)
                senders.append(j)

    #print(f'F_symreg: {F_symreg.shape}')
    #print(f'Fn_symreg: {Fn_symreg.shape}')

    # Flatten the arrays
    D_V_tr = np.reshape(D_V_tr, [-1, 6])

    # Convert them to tensors
    D_V_tr = tf.convert_to_tensor(D_V_tr, dtype="float32")

    # Create a normalization layer
    norm_layer = Normalize_gn(hf.cartesian_to_spherical_coordinates(D_V_tr))
    return D_V_symreg, F_symreg, Fn_symreg, norm_layer, senders, receivers


def run_symbolic_regression(D_V, Fn_symreg, model, system, num_pts=1000, name='eqns'):
    D_V_tf = tf.convert_to_tensor(D_V.reshape(-1, 6), dtype="float32")
    _, F = model.call(D_V_tf, extract=True)

    names = system.get_names()
    learned_masses = model.logm_planets.numpy()
    masses = system.get_masses()
    isun = names.index("Sun")
    learned_msun = learned_masses[isun]
    learned_masses -= learned_msun

    F_corr = (F / Fn_symreg) - 1

    X = np.zeros([D_V.shape[0], D_V.shape[1], 11])
    # Distances
    X[:, :, 2:5] = D_V[:, :, :3]
    # Distances Norm
    X[:, :, 5] = np.linalg.norm(D_V[:, :, :3], axis=-1)
    # Velocities
    X[:, :, 6:9] = D_V[:, :, 3:]
    # Velocities norm
    X[:, :, 9] = np.linalg.norm(D_V[:, :, 3:], axis=-1)

    k = 0
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                X[:, k, 0] = 10 ** (learned_masses[i])
                X[:, k, 1] = 10 ** (learned_masses[j])
                k += 1

    X = X.reshape([-1, 11])
    F = F.reshape([-1, 3])
    F_corr = F_corr.reshape([-1, 3])
    y = F[:, 0]  # F_x
    y_corr = F_corr[:, 0]
    y /= np.std(y)  # same as above
    y_corr /= np.std(y_corr)

    m_std = np.std(X[:, :2])
    x_std = np.std(X[:, 2:5])
    v_std = np.std(X[:, 6:9])
    #l_std = np.std(X[:, 10])
    X[:, :2] /= m_std
    X[:, 2:6] /= x_std
    X[:, 6:10] /= v_std
    #X[:, 10] /= l_std
    # Angular momentum norm
    X[:, 10] = np.linalg.norm(np.cross(X[:, 2:5], X[:, 6:9]))

    XV = np.zeros([D_V.shape[0], 5])
    XV[:, :2] = X[:, :2]    # m
    XV[:, 2] = X[:, 2]     # x
    XV[:, 3] = X[:, 5]     # r
    XV[:, 4] = X[:, 10]    # l

    idx = np.random.choice(X.shape[0], num_pts, replace=False)
    X = X[idx]
    y = y[idx]
    y_corr = y_corr[idx]
    XV = XV[idx]

    # Randomly swap masses
    for i in range(len(X)):
        r = np.random.rand()
        if r > 0.5:
            X[i, 0], X[i, 1] = X[i, 1], X[i, 0]

    pysr_model = PySRRegressor(populations=64,
                               niterations=1000,
                               binary_operators=["plus", "sub", "mult", "div",
                                                 #"cross(x, y) = cross(x,y)"
                                                 ],
                               unary_operators=["square", "cube", "quad(x) = x^4", "quint(x) = x^5",
                                                #"norm(x) = norm(x, 2)"
                                                ],
                               constraints={
                                   "div": (-1, 9),
                                   "square": 9,
                                   "cube": 9,
                               },
                               temp_equation_file=False,
                               equation_file=os.path.join(
                                   "./data/saved_equations/", name),
                               batching=True,
                               batch_size=50,
                               progress=False,
                               procs=4,
                               annealing=False,
                               maxsize=40,
                               maxdepth=10,  # avoid deep nesting
                               useFrequency=True,
                               optimizer_algorithm="BFGS",
                               optimizer_iterations=10,
                               optimize_probability=1.0
                               )

    pysr_model.fit(X, y)
    return pysr_model


if __name__ == "__main__":
    tf.config.list_physical_devices('CPU')
    tf.config.run_functions_eagerly(False)

    data_tr, _, data_symreg, system = read_data(num_time_steps_tr,
                                                num_time_steps_val)
    print('Read data')
    D_V_symreg, F_symreg, Fn_symreg, norm_layer, senders, receivers = format_data_symreg(
        data_tr, data_symreg, system)
    print('Formatted data')
    model = load_model(system, norm_layer, senders, receivers)
    print('Model loading completed')

    equations = run_symbolic_regression(D_V_symreg, Fn_symreg, model, system,
                                        name='sun_mercury_venus_1000_gr_30y_1_xy',
                                        num_pts=5000)

    #best(equations)
