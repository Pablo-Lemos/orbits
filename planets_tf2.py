#!/usr/bin/env python
# coding: utf-8

# Imports
import sys

from ml_model import *
import pickle
import os
from simulator import base_classes

sys.modules['base_classes'] = base_classes
print('Started')

# Global constants
AU = 149.6e6 * 1000 # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY #Year
delta_time = (0.5/24.) # 2 hours
MSUN = 1.9885e+30
MEARTH = 5.9724e+24
G = 6.67428e-11/AU**3*MSUN*DAY**2

# Training variables
patience = 5
d_patience = 0
noise_level = 0.01
log_every_iterations = 1000
num_training_iterations = 200000
num_time_steps_tr = 130000  # An orbit for saturn is 129110 steps

def force_newton(x, m1, m2):
    return G*m1*m2/np.linalg.norm(x, axis=-1, keepdims=True)**3.*x


def read_data(num_time_steps_tr):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #filename = os.path.join(dir_path, 'data/solar_system_data.pkl')
    filename = os.path.join(dir_path, 'data/debug.pkl')
    filehandler = open(filename, 'rb')
    system = pickle.load(filehandler)
    x = system.get_positions()
    v = system.get_velocities()
    data = np.concatenate([x, v], axis=-1)

    # Get the acceleration
    A = data[1:, :, 3:] - data[:-1, :, 3:]
    data[:-1, :, 3:] = A / delta_time
    data = data[:-1]

    # For debugging, reduce size of validation data. Really speeds things up!
    nval = 10000
    data = data[:(num_time_steps_tr + nval)]

    #masses = system.get_masses()/MSUN

    # Split into training and validation
    data_tr = data[:num_time_steps_tr]
    data_val = data[num_time_steps_tr:]

    # Shuffle the data
    np.random.shuffle(data_tr)
    np.random.shuffle(data_val)

    data_tr = data_tr[:num_time_steps_tr]
    return data_tr, data_val, system


def format_data(data_tr, data_val, system):
    num_time_steps_tr = len(data_tr)
    num_time_steps_val = len(data_val)

    # Do not change this
    batch_size_tr = 16
    num_batches = num_time_steps_tr // batch_size_tr

    nedges = system.numEdges
    D_tr_np = np.empty([len(data_tr), nedges, 3])
    D_val_np = np.empty([len(data_val), nedges, 3])
    #F_val = np.empty([len(data_val), nedges, 3])
    k = 0
    #names_edges = []
    senders, receivers = [], []
    for i in range(system.numPlanets):
        for j in range(system.numPlanets):
            if i > j:
                d_tr = data_tr[:, j, :3] - data_tr[:, i, :3]
                d_val = data_val[:, j, :3] - data_val[:, i, :3]
                D_tr_np[:, k, :] = d_tr
                D_val_np[:, k, :] = d_val
                #F_val[:, k, :] = force_newton(d_val, masses[i], masses[
                    #j])  # cartesian_to_spherical_coordinates(d_val)
                #names_edges.append(names[j] + ' - ' + names[i])

                k += 1
                receivers.append(i)
                senders.append(j)

    A_tr = data_tr[:, :, 3:]
    A_val = data_val[:, :, 3:]
    A_norm = np.std(A_tr)

    D_tr_flat = np.reshape(D_tr_np, [-1, 3])
    D_val_flat = np.reshape(D_val_np, [1, -1, 3])

    A_tr_flat = np.reshape(A_tr / A_norm, [-1, 3])
    A_val_flat = np.reshape(A_val / A_norm, [1, -1, 3])

    D_tr = tf.convert_to_tensor(D_tr_flat, dtype="float32")
    A_tr = tf.convert_to_tensor(A_tr_flat, dtype="float32")

    D_tr_batches = tf.split(D_tr, num_batches)
    A_tr_batches = tf.split(A_tr, num_batches)

    D_val = tf.convert_to_tensor(D_val_flat, dtype="float32")
    A_val = tf.convert_to_tensor(A_val_flat, dtype="float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (D_tr_batches, A_tr_batches))

    test_ds = tf.data.Dataset.from_tensor_slices(
        (D_val, A_val))

    norm_layer = Normalize_gn(cartesian_to_spherical_coordinates(D_tr))
    return train_ds, test_ds, norm_layer, senders, receivers

def main(system, train_ds, test_ds, norm_layer, senders, receivers):
    checkpoint_filepath = './saved_models/orbits'

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      verbose=1,
                                                      patience=20,
                                                      restore_best_weights=False)
    #  Restore best weights not working, but found way around using checkpoint

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=True,
        verbose=0)
    model = LearnForces(system.numPlanets, senders, receivers, norm_layer,
                        noise_level=noise_level)

    # model.compile(run_eagerly=True)
    model.compile()

    model.fit(train_ds,
              epochs=1000,
              verbose=2,
              callbacks=[early_stopping, checkpoint],
              validation_data=test_ds
              )

    model.load_weights(checkpoint_filepath)

    return model


if __name__ == "__main__":
    tf.config.list_physical_devices('CPU')
    tf.config.run_functions_eagerly(False)
    nplanets = 8  #  Number of planets (not counting the sun)

    data_tr, data_val, system = read_data(num_time_steps_tr)
    print('Read data')
    train_ds, test_ds, norm_layer, senders, receivers = format_data(data_tr, data_val, system)
    print('Formatted data')
    model = main(system, train_ds, test_ds, norm_layer, senders, receivers)
    print('Model training completed')

    print ('Learned planetary masses:')
    print(np.round(model.logm_planets.numpy(), 2))

    print ('True planetary masses:')
    print(np.round(np.log10(masses)[1:], 2))





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