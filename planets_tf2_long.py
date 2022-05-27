#!/usr/bin/env python
# coding: utf-8

# Imports
import sys

from ml_model_long import *
import pickle
import os
from simulator import base_classes

sys.modules['base_classes'] = base_classes
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
patience = 100 # For early stopping
noise_level = 0.01 # Standard deviation of Gaussian noise for randomly perturbing input data
num_epochs = 1000 # Number of training epochs. Set to large number
num_time_steps_tr = 8000  # Number of time steps for training (~27 years).
# One time step is 30 minutes
# An orbit for saturn is 129110 steps
num_time_steps_val = 80 # Using few to speed up calculations

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
    filename = os.path.join(dir_path, 'data/reduced.pkl')
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
    data = data[:(num_time_steps_tr + num_time_steps_val)]

    # Split into training and validation
    data_tr = data[:num_time_steps_tr]
    data_val = data[num_time_steps_tr:]

    # Shuffle the data
    np.random.shuffle(data_tr)
    np.random.shuffle(data_val)

    return data_tr, data_val, system


def format_data(data_tr, data_val, system):
    """
    Convert the data into normalized tensorflow data objects that we can
    use for training
    """

    num_time_steps_tr = len(data_tr)

    # Do not change this
    batch_size_tr = 8
    num_batches = num_time_steps_tr // batch_size_tr

    nedges = system.numEdges

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
    return train_ds, test_ds, norm_layer, senders, receivers


def main(system, train_ds, test_ds, norm_layer, senders, receivers):
    """ Main function: Create model and train"""

    # Create my callbacks: early stopping and checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      verbose=1,
                                                      patience=patience,
                                                      restore_best_weights=False)

    # Restore best weights not working, but found way around using checkpoint
    checkpoint_filepath = './saved_models/orbits'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=True,
        verbose=0)

    # Create a model
    model = LearnForces(system.numPlanets, senders, receivers, norm_layer,
                        noise_level=noise_level, masses=system.get_masses())

    # Compile
    #model.compile(run_eagerly=True)
    model.compile()
    #model.evaluate(train_ds)

    #model.load_weights(
    #    os.path.join(checkpoint_filepath, 'reduced_weights.ckpt'))

    #model.initiate_weights(test_ds)
    #model.save_weights(
    #    os.path.join(checkpoint_filepath, 'initial_weights.ckpt'))
    # w = model.flatten_weights()
    # print(w)
    # w2 = tf.ones_like(w)
    #
    # model.load_weights(w2)

    #print(model.trainable_variables)

    model.fit(train_ds,
              epochs=num_epochs,
              verbose=2,
              callbacks=[early_stopping, checkpoint],
              validation_data=test_ds
              )

    #model.load_weights(checkpoint_filepath)
    model.save_weights(
        os.path.join(checkpoint_filepath, 'reduced_weights.ckpt'))


    return model


if __name__ == "__main__":
    tf.config.list_physical_devices('CPU')
    tf.config.run_functions_eagerly(False)

    data_tr, data_val, system = read_data(num_time_steps_tr, num_time_steps_val)
    print('Read data')
    train_ds, test_ds, norm_layer, senders, receivers = format_data(data_tr, data_val, system)
    print('Formatted data')
    model = main(system, train_ds, test_ds, norm_layer, senders, receivers)
    print('Model training completed')

    for mass, true_mass, name in zip(model.logm_planets.numpy(), system.get_masses(), names):
        print(name, np.round(np.log10(true_mass + 1e-22) + 10, 2), np.round(mass, 2))





# Symbolic Regression
'''

dv = tf.reshape(D_val, [-1, nedges, 3])
dv, fp = rotate_data(dv, fp)
F_pred_sr = np.empty([num_time_steps_sr, nedges, 3])
D_val_sr = np.empty([num_time_steps_sr, nedges, 3])
for i in range(num_time_steps_sr):
    Dv_temp, Fp_temp = rotate_data(dv[i], fp[i])
    F_pred_sr[i] = Fp_temp
    D_val_sr[i] = Dv_temp


X = np.zeros([nedges*num_time_steps_sr,6])
F = np.zeros([nedges*num_time_steps_sr,3])
weights = np.zeros(nedges*num_time_steps_sr)
k=0
masses_learned = np.zeros(nplanets)
masses_learned[0] = masses[0]
masses_learned[1:] = 10**model.logm_planets.numpy()
for i in range(nplanets):
    for j in range(nplanets):
        if i>j:
            #X[k*num_time_steps_sr:(k+1)*num_time_steps_sr,0] = masses[i]
            #X[k*num_time_steps_sr:(k+1)*num_time_steps_sr,1] = masses[j]
            X[k*num_time_steps_sr:(k+1)*num_time_steps_sr,0] = np.log(masses_learned[i])
            X[k*num_time_steps_sr:(k+1)*num_time_steps_sr,1] = np.log(masses_learned[j])
            X[k*num_time_steps_sr:(k+1)*num_time_steps_sr,2:5] = D_val_sr[:,k,:]
            X[k*num_time_steps_sr:(k+1)*num_time_steps_sr,5] = np.linalg.norm(D_val_sr[:,k,:], axis = -1)#**3
            F[k*num_time_steps_sr:(k+1)*num_time_steps_sr,:] = F_pred_sr[:,k,:]#/F_norm #works better with
            invw = np.mean(np.linalg.norm(F_pred_sr[:,k,:], axis = -1))
            if invw <1e-50:
                weights[k*num_time_steps_sr:(k+1)*num_time_steps_sr] = 1e-100
            else: 
                weights[k*num_time_steps_sr:(k+1)*num_time_steps_sr] = -np.log10(invw)
            k+=1

weights/=max(weights)

from pysr import pysr
# Learn equations
equations = []
for i in range(1):
    equation = pysr(X[:,:], F[:,i], niterations=10,
                    #batching = True, 
                    #batchSize = 500,
            weights = weights,      
            #maxsize = 100,
            populations = 4,
            variable_names = ['m0', 'm1', 'x', 'y', 'z', 'r'],
            binary_operators=["mult", "div"],
            unary_operators=["square", "cube"],            
            #binary_operators=["plus", "sub", "mult", "div"],
            #unary_operators=["square", "cube", "exp", "logm", "logm10"],
                   )
    equations.append(equation)

'''