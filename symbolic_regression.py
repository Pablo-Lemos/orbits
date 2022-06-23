from planets_tf2 import read_data
from ml_model import LearnForces, Normalize_gn
from helper_functions import cartesian_to_spherical_coordinates
import tensorflow as tf
import numpy as np
from pysr import PySRRegressor
import os

# Training variables
num_time_steps_tr = 512000  # Number of time steps for training (~27 years).
noise_level = 0.01 # Standard deviation of Gaussian noise for randomly perturbing input data
# One time step is 30 minutes
# An orbit for saturn is 129110 steps
num_time_steps_val = 10000 # Using few to speed up calculations

# Global constants
AU = 149.6e6 * 1000 # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY # Year
delta_time = (0.5/24.) # 30 minutes
MSUN = 1.9885e+30 # kg
MEARTH = 5.9724e+24 # kg
G = 6.67428e-11/AU**3*MSUN*DAY**2 # Change units of G to AU^3 MSun^{-1} Day^{-2}


def force_newton(x, m1, m2):
    return G*m1*m2/np.linalg.norm(x, axis = -1, keepdims=True)**3.*x


def load_model(system, norm_layer, senders, receivers):
    """ Load the model"""

    # Restore best weights not working, but found way around using checkpoint
    checkpoint_filepath = './saved_models/planetsonly_i2'

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

    nedges = system.numEdges
    masses = system.get_masses()

    # Create empty arrays for the distances for training and validation
    D_tr = np.empty([len(data_tr), nedges, 3])
    D_symreg = np.empty([len(data_symreg), nedges, 3])
    F_symreg = np.empty([len(data_symreg), nedges, 3])

    k = 0
    # Create empty lists for the senders and receivers that will be used for
    # the edges of the graph
    senders, receivers = [], []
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                # For every pair of objects, assign a distance
                D_tr[:, k, :] = data_tr[:, j, :3] - data_tr[:, i, :3]
                D_symreg[:, k, :] = data_symreg[:, j, :3] - \
                                    data_symreg[:, i, :3]
                F_symreg[:, k, :] = force_newton(
                    D_symreg[:, k, :], masses[i], masses[j])

                k += 1
                # Add sender and receiver index
                receivers.append(i)
                senders.append(j)

    # Flatten the arrays
    D_tr = np.reshape(D_tr, [-1, 3])

    # Convert them to tensors
    D_tr = tf.convert_to_tensor(D_tr, dtype="float32")

    # Create a normalization layer
    norm_layer = Normalize_gn(cartesian_to_spherical_coordinates(D_tr))
    return D_symreg, F_symreg, norm_layer, senders, receivers


def run_symbolic_regression(D, model, system, num_pts=1000, name='eqns'):
    D_tf = tf.convert_to_tensor(D.reshape(-1, 3), dtype="float32")
    _, F = model.call(D_tf, extract=True)

    names = system.get_names()
    learned_masses = model.logm_planets.numpy()
    isun = names.index("sun")
    learned_msun = learned_masses[isun]
    learned_masses /= learned_msun

    X = np.zeros([D.shape[0], D.shape[1], 6])
    X[:, :, 2:5] = D
    X[:, :, 5] = np.linalg.norm(D, axis=-1)
    k=0
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                X[:, k, 0] = 10 ** (learned_masses[i])
                X[:, k, 1] = 10 ** (learned_masses[j])
                # print(fp[0,k,0], X[0,k,0]*X[0,k,1]*X[0,k,2]/X[0,k,5]**3.)
                k += 1

    X = X.reshape([-1, 6])
    F = F.reshape([-1, 3])
    y = F[:,0] #F_x
    #X[:, [0, 1]] = np.exp(X[:, [0, 1]])/1e23 #re-scale to prevent precision issues, since pysr uses 32-bit floats
    y /= np.std(y)                                 #same as above

    m_std = np.std(X[:,:2])
    x_std = np.std(X[:,2:5])
    X[:,:2]/= m_std
    X[:,2:]/= x_std

    idx = np.random.choice(X.shape[0], num_pts, replace=False)
    X = X[idx]
    y = y[idx]


    pysr_model = PySRRegressor(populations=64,
                               binary_operators=["plus", "sub", "mult",
                                                 "pow", "div"],
                               unary_operators=["neg", "exp", "log_abs",
                                                "sin", "cos"],
                               temp_equation_file=False,
                               equation_file=os.path.join(
                                   "./data/saved_equations/", name),
                               progress=False,
                               procs=4,
                               annealing=False,
                               maxsize=40,
                               useFrequency=True,
                               variable_names = ['m0', 'm1', 'x', 'y', 'z', 'r'],
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
    D_symreg, F_symreg, norm_layer, senders, receivers = format_data_symreg(
        data_tr, data_symreg, system)
    print('Formatted data')
    model = load_model(system, norm_layer, senders, receivers)
    print('Model loading completed')

    equations = run_symbolic_regression(D_symreg, model, system,
                                        name="planetsonly")


