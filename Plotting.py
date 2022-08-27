from planets_tf2 import read_data
from ml_model import LearnForces, Normalize_gn
from helper_functions import cartesian_to_spherical_coordinates
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


# Training variables
num_time_steps_tr = 1031200  # Number of time steps for training (~27 years).
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


def force_newton(x, m1, m2):
    return G*m1*m2/np.linalg.norm(x, axis=-1, keepdims=True)**3.*x


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
    filename = './simulator/newton_simulation_1pl_60y.pickle'
    filehandler = open(filename, 'rb')
    system = pickle.load(filehandler)

    # Extract the position and velocity
    x = system.get_positions()
    v = system.get_velocities()

    # Concatenate them into a numpy array
    # Data has three dimensions: Time, body, and a length 6 dimension with x and v
    data = np.concatenate([x, v], axis=-1)

    # Convert velocity into acceleration
    A = data[1:, :, 3:] - data[:-1, :, 3:]
    data[:-1, :, 3:] = A / delta_time
    data = data[:-1]

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

    # Create empty arrays for the distances for training and validation
    D_tr = np.empty([len(data_tr), nedges, 3])
    D_val = np.empty([len(data_val), nedges, 3])

    k = 0
    # Create empty lists for the senders and receivers that will be used for
    # the edges of the graph
    senders, receivers = [], []
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                # For every pair of objects, assign a distance
                D_tr[:, k, :] = data_tr[:, j, :3] - data_tr[:, i, :3]
                D_val[:, k, :] = data_val[:, j, :3] - data_val[:, i, :3]

                k += 1
                # Add sender and receiver index
                receivers.append(i)
                senders.append(j)

    # Accelerations
    A_tr = data_tr[:, :, 3:]
    A_val = data_val[:, :, 3:]
    # Normalization of the accelerations
    A_norm = np.std(A_tr)

    # Flatten the arrays
    D_tr = np.reshape(D_tr, [-1, 3])
    D_val = np.reshape(D_val, [1, -1, 3])

    A_tr = np.reshape(A_tr / A_norm, [-1, 3])
    A_val = np.reshape(A_val / A_norm, [1, -1, 3])

    # Convert them to tensors
    D_tr = tf.convert_to_tensor(D_tr, dtype="float32")
    A_tr = tf.convert_to_tensor(A_tr, dtype="float32")

    D_val = tf.convert_to_tensor(D_val, dtype="float32")
    A_val = tf.convert_to_tensor(A_val, dtype="float32")

    # Split the training arrays into batches
    D_tr_batches = tf.split(D_tr, num_batches)
    A_tr_batches = tf.split(A_tr, num_batches)

    # Convert into tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (D_tr_batches, A_tr_batches))

    test_ds = tf.data.Dataset.from_tensor_slices(
        (D_val, A_val))

    # Create a normalization layer
    norm_layer = Normalize_gn(cartesian_to_spherical_coordinates(D_tr))

    return train_ds, test_ds, norm_layer, senders, receivers, A_norm, D_val


def load_model(system, norm_layer, senders, receivers):
    """ Load the model"""

    # Restore best weights not working, but found way around using checkpoint
    checkpoint_filepath = './saved_models/sun_mercury_n_60y_2'

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

    data_tr, data_val, _, system = read_data(num_time_steps_tr,
                                           num_time_steps_val)
    print('Read data')
    train_ds, test_ds, norm_layer, senders, receivers, A_norm, D_val = format_data_gnn(
        data_tr, data_val, system)
    print('Formatted data')
    model = load_model(system, norm_layer, senders, receivers)
    print('Model loading completed')

    # Evaluating on the validation data
    ap, fp = model(D_val[0], extract=True)

    nedges = system.numEdges
    masses = system.get_masses()
    nplanets = system.numPlanets
    learned_masses = model.logm_planets.numpy()
    names = system.get_names()

    F_val_new = np.empty([len(data_val), nedges, 3])
    k = 0
    for i in range(nplanets):
        for j in range(nplanets):
            if i > j:
                d_val = data_val[:, j, :3] - data_val[:, i, :3]
                print(f'd_val shape: {d_val.shape}')
                F_val_new[:, k, :] = force_newton(d_val, 10 ** learned_masses[i], 10 ** learned_masses[j])  # cartesian_to_spherical_coordinates(d_val)
                k += 1

    a_diff = np.zeros([data_val.shape[0], data_val.shape[1], 3])
    # acc norm
    at = (data_val[:, :, 3:] / A_norm)
    # Resultant true acc
    a_res_true = np.linalg.norm(at, axis=-1)
    # Resultant predicted acc
    a_res_pre = np.linalg.norm(ap, axis=-1)
    a_diff = a_res_pre - a_res_true
    # Number of Validation data points
    n = a_res_pre.shape[0]
    # Standard deviation
    a_diff_2 = np.sum((a_diff) ** 2, axis=0)
    sigma = (a_diff_2 / n) ** 0.5

    x = at - ap
    x = np.sum(x ** 2, axis=-1)
    x = np.sum(x, axis=0)

    x2 = at
    x2 = np.sum(x2 ** 2, axis=-1)
    x2 = np.sum(x2, axis=0)

    loss = np.sum(x / x2)


    # nrows = nplanets // 4
    nrows = 1
    fig, ax = plt.subplots(nrows, 2, figsize=(16, 8 * nrows))
    fig.suptitle(r'Newtonian Trained Model', fontsize=18)

    for i in range(nplanets):
        ax[i // 1].set_title(names[i], fontsize=16)
        ax[i // 1].set_xlabel(r'$a_x$', fontsize=12)
        ax[i // 1].set_ylabel(r'$a_y$', fontsize=12)
        ax[i // 1].plot(data_val[:, i, 3] / A_norm, data_val[:, i, 4] / A_norm, '.', label='Truth')
        ax[i // 1].plot(ap[:, i, 0], ap[:, i, 1], '.', label='Learned')


    ax[0].legend()

    '''
    # Histogram
    for i in range(nplanets):
        ax[i // 1].set_title(names[i], fontsize=16)
        ax[i // 1].set_xlabel(r'$\delta$$a_{r}^{2}$', fontsize=12)
        ax[i // 1].set_ylabel(r'Log(N)')
        ax[i // 1].hist(a_diff[:, i]**2, bins=250, color='limegreen', log=True)
        if i == 0:
            ax[i // 1].text(1e-18, 1700, r'$\mu_{Sun}=5.53\times10^{-18}$', fontsize=12)
            ax[i // 1].text(1e-18, 1050, r'$\sigma_{Sun}=7.02\times10^{-18}$', fontsize=12)
        else:
            ax[i // 1].text(0.00005, 1400, r'$\mu_{Mercury}=1.12\times10^{-4}$', fontsize=12)
            ax[i // 1].text(0.00005, 1000, r'$\sigma_{Mercury}=1.37\times10^{-4}$', fontsize=12)
    '''
    #plt.hist(np.mean(a_diff[:, 1, :], axis=-1), bins=300)

    #plt.hist(loss[:, 1], bins=300)

    std_sun = np.std(a_diff[:, 0]**2)
    std_mercury = np.std(a_diff[:, 1]**2)
    mean_sun = np.mean(a_diff[:, 0]**2)
    mean_mercury = np.mean(a_diff[:, 1]**2)
    print(f'Std Sun: {std_sun}')
    print(f'Std Mercury: {std_mercury}')
    print(f'Mean Sun: {mean_sun}')
    print(f'Mean Mercury: {mean_mercury}')
    print(f'Loss: {loss}')
    print(f'Loss mean: {np.mean(loss)}')
    print(f'Loss std: {np.std(loss)}')
    print(f'Loss Min: {loss.min()}, Loss Max: {loss.max()}')
    print(f'diff Min: {a_diff.min()}, diff Max: {a_diff.max()}')
    print(f'x Min: {x.min()}, x Max: {x.max()}')
    print(f'x2 Min: {x2.min()}, x2 Max: {x2.max()}')
    print(f'Sigma: {sigma}')

    plt.show()