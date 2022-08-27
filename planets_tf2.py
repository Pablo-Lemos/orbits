#!/usr/bin/env python
# coding: utf-8

# Imports
import sys

from ml_model import *
import pickle
import os
from simulator import base_classes_GR

sys.modules['base_classes_GR'] = base_classes_GR
print('Started')

# Global constants
AU = 149.6e6 * 1000 # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY # Year
delta_time = (0.5/24.) # 30 minutes
MSUN = 1.9885e+30 # kg
MEARTH = 5.9724e+24 # kg
G = 6.67428e-11/AU**3*MSUN*DAY**2 # Change units of G to AU^3 MSun^{-1} Day^{-2}

# Training variables
patience = 50 # For early stopping
noise_level = 0.01 # Standard deviation of Gaussian noise for randomly perturbing input data
num_epochs = 1000 # Number of training epochs. Set to large number
num_time_steps_tr = 1031200  # Number of time steps for training (~28.8 years).
# One time step is 30 minutes
# An orbit for saturn is 129110 steps
num_time_steps_val = 10000 # Using few to speed up calculations
num_time_steps_symreg = 10000 # Using few to speed up calculations


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
    return train_ds, test_ds, norm_layer, senders, receivers, A_norm


def main(system, train_ds, test_ds, norm_layer, senders, receivers):
    """ Main function: Create model and train"""

    # Create my callbacks: early stopping and checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      verbose=1,
                                                      patience=patience,
                                                      restore_best_weights=False)

    # Restore best weights not working, but found way around using checkpoint
    checkpoint_filepath = './saved_models/sun_mercury_n_60y_2'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=True,
        verbose=0)

    # Create a model
    model = LearnForces(system.numPlanets, senders, receivers, norm_layer,
                        noise_level=noise_level)

    # Compile
    # model.compile(run_eagerly=True)
    model.compile()

    model.fit(train_ds,
              epochs=num_epochs,
              verbose=2,
              callbacks=[early_stopping, checkpoint],
              validation_data=test_ds
              )

    model.load_weights(checkpoint_filepath)

    return model


if __name__ == "__main__":
    #tf.config.list_physical_devices('CPU')
    tf.config.run_functions_eagerly(False)

    data_tr, data_val, _, system = read_data(num_time_steps_tr,
                                           num_time_steps_val)
    print('Read data')
    train_ds, test_ds, norm_layer, senders, receivers, A_norm = format_data_gnn(
        data_tr, data_val, system)
    print('Formatted data')
    print('A_norm =', A_norm)
    model = main(system, train_ds, test_ds, norm_layer, senders, receivers)
    print('Model training completed')

    print('Learned planetary masses:')
    names = system.get_names()
    true_masses = system.get_masses()
    learned_masses = model.logm_planets.numpy()
    isun = names.index("Sun")
    true_msun = true_masses[isun]
    learned_msun = learned_masses[isun]
    for learned_mass, true_mass, name in zip(
            learned_masses, true_masses, names):
        print(name, np.round(np.log10(true_mass/true_msun), 2),
              np.round(learned_mass - learned_msun, 2))