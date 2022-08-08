from planets_tf2_gr import read_data
from ml_model_gr import LearnForces, Normalize_gn
import helper_functions_gr as hf
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


# Training variables
num_time_steps_tr = 1031200  # Number of time steps for training 30y=504000; 40y=680800; 50y=856000; 60y=1031200
noise_level = 0.01 # Standard deviation of Gaussian noise for randomly perturbing input data
# One time step is 30 minutes
# An orbit for saturn is 129110 steps
num_time_steps_val = 10000 # Using few to speed up calculations
num_time_steps_symreg = 10000 # Using few to speed up calculations

# Global constants
AU = 149.6e6 * 1000 # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY # Year
delta_time = (0.5/24.) # 30 minutes
MSUN = 1.9885e+30 # kg
MEARTH = 5.9724e+24 # kg
G = 6.67428e-11/AU**3*MSUN*DAY**2 # Change units of G to AU^3 MSun^{-1} Day^{-2}
A_norm = 0.00042411583592113497 # From planets_tf2 (I will change the way
# this is stored eventually)
c = (2.99792458 * 10**8) * DAY / AU #speed of light in AU/Day


def force_newton(x, m1, m2):
    return G*m1*m2/np.linalg.norm(x, axis=-1, keepdims=True)**3.*x

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
    corr = 3*L_norm**2/(c**2 * dist_norm**2)
    return f_n * (1 + 1000 * corr)


def read_data(num_time_steps_tr, num_time_steps_val):
    """
    Read the data
    :param num_time_steps_tr: Size of training set
    :param num_time_steps_val: Size of validation set
    :return: training data, validation data, and a Star System object
    containing three dimensions: Time, body, and a length 6 dimension with
    x and v
    """

    # Read the file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #filename = os.path.join(dir_path, 'data/solar_system_data.pkl')
    filename = './simulator/1000_gr_simulation_1pl_60y.pickle'
    filehandler = open(filename, 'rb')
    system = pickle.load(filehandler)

    # Extract the position and velocity
    x = system.get_positions()
    v = system.get_velocities()

    # Calculate acceleration
    A = v[1:, :] - v[-1, :]
    a = A / delta_time

    # Remove last row from x & v to match the shape of a
    x = x[:-1]
    v = v[:-1]

    # Concatenate them into a numpy array
    # Data has four dimensions: Time, body, and 3 dimensions each for x, v, and a
    data = np.concatenate([x, v, a], axis=-1)

    # Eliminate unused data
    data = data[:(num_time_steps_tr + num_time_steps_val + num_time_steps_symreg)]

    # Split into training and validation
    data_tr = data[:num_time_steps_tr]
    data_val = data[num_time_steps_tr:num_time_steps_tr+num_time_steps_val]
    data_symreg = data[-num_time_steps_symreg:]

    # Shuffle the data
    np.random.shuffle(data_tr)
    np.random.shuffle(data_val)
    np.random.shuffle(data_symreg)

    return data_tr, data_val, data_symreg, system


def format_data_gnn(data_tr, data_val, system):
    """
    Convert the data into normalized tensorflow data objects that we can
    use for training
    """

    num_time_steps_tr = len(data_tr)

    # Do not change this
    batch_size_tr = 16
    num_batches = num_time_steps_tr // batch_size_tr

    nedges = system.numEdges
    masses = system.get_masses()

    # Create empty arrays for the distances and velocities for training and validation
    D_V_tr = np.empty([len(data_tr), nedges, 6])
    D_V_val = np.empty([len(data_val), nedges, 6])

    k = 0
    # Create empty lists for the senders and receivers that will be used for
    # the edges of the graph
    senders, receivers = [], []
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                # For every pair of objects, assign a distance
                D_V_tr[:, k, :] = data_tr[:, j, :6] - data_tr[:, i, :6]
                D_V_val[:, k, :] = data_val[:, j, :6] - data_val[:, i, :6]

                k += 1
                # Add sender and receiver index
                receivers.append(i)
                senders.append(j)

    # Accelerations
    A_tr = data_tr[:, :, 6:]
    A_val = data_val[:, :, 6:]
    # Normalization of the accelerations
    A_norm = np.std(A_tr)

    # Flatten the arrays
    D_V_tr = np.reshape(D_V_tr, [-1, 6])
    D_V_val = np.reshape(D_V_val, [1, -1, 6])

    A_tr = np.reshape(A_tr / A_norm, [-1, 3])
    A_val = np.reshape(A_val / A_norm, [1, -1, 3])

    # Convert them to tensors
    D_V_tr = tf.convert_to_tensor(D_V_tr, dtype="float32")
    A_tr = tf.convert_to_tensor(A_tr, dtype="float32")

    D_V_val = tf.convert_to_tensor(D_V_val, dtype="float32")
    A_val = tf.convert_to_tensor(A_val, dtype="float32")

    # Split the training arrays into batches
    D_V_tr_batches = tf.split(D_V_tr, num_batches)
    A_tr_batches = tf.split(A_tr, num_batches)

    # Convert into tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (D_V_tr_batches, A_tr_batches))

    test_ds = tf.data.Dataset.from_tensor_slices(
        (D_V_val, A_val))

    # Create a normalization layer (using the gr helper function)
    norm_layer = Normalize_gn(hf.cartesian_to_spherical_coordinates(D_V_tr))
    return train_ds, test_ds, norm_layer, senders, receivers, A_norm, D_V_val


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


if __name__ == "__main__":
    tf.config.list_physical_devices('CPU')
    tf.config.run_functions_eagerly(False)

    data_tr, data_val, _, system = read_data(num_time_steps_tr, num_time_steps_val)
    print('Read data')
    train_ds, test_ds, norm_layer, senders, receivers, A_norm, D_V_val = format_data_gnn(
        data_tr, data_val, system)
    print('Formatted data')
    model = load_model(system, norm_layer, senders, receivers)
    print('Model loading completed')

    # Evaluating on the validation data
    ap, fp = model(D_V_val[0], extract=True)

    nedges = system.numEdges
    masses = system.get_masses()
    nplanets = system.numPlanets
    learned_masses = model.logm_planets.numpy()
    names = system.get_names()

    F_val_new = np.empty([len(data_val), nedges, 3])
    d_val = np.empty([len(data_val), nedges, 6])

    k = 0
    for i in range(nplanets):
        for j in range(nplanets):
            if i > j:
                d_val = data_val[:, j, :6] - data_val[:, i, :6]
                F_val_new[:, k, :] = GR_correctoin(10 ** learned_masses[i], 10 ** learned_masses[j],
                                                   d_val[:, :3], d_val[:, 3:6] ) # cartesian_to_spherical_coordinates(d_val)
                k += 1

    nrows = nplanets // 2
    fig, ax = plt.subplots(nrows, 2, figsize=(16, 8 * nrows))
    for i in range(nplanets):
        ax[i // 1].set_title(names[i], fontsize=16)
        ax[i // 1].plot(ap[:, i, 0], ap[:, i, 1], '.', label='Learned')
        ax[i // 1].plot(data_val[:, i, 6] / A_norm, data_val[:, i, 7] / A_norm, '.', label='Truth')

    ax[0].legend()
    plt.show()
