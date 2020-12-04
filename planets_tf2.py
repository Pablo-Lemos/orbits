#!/usr/bin/env python
# coding: utf-8

# Imports
import numpy as np
import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import matplotlib.pyplot as plt
import math
import gc

import read_orbits
from solar_system_names import *
from ml_model import *

print('Started')

# Global constants
AU = 149.6e6 * 1000 # Astronomical Unit in meters.
DAY = 24*3600. # Day in seconds
YEAR = 365.25*DAY #Year
delta_time = (0.5/24.) # 2 hours
MSUN = 1.9885e+30
MEARTH = 5.9724e+24
G = 6.67428e-11/AU**3*MEARTH*DAY**2

def force_newton(x, m1, m2):
    return G*m1*m2/np.linalg.norm(x, axis = -1, keepdims=True)**3.*x


# Training variables
patience = 5
d_patience = 0
noise_level = 0.01
log_every_iterations = 1000
num_training_iterations = 200000

# Do not change this
#total_time_traj = 35 #Years
#num_time_steps_total = int(total_time_traj/delta_time)
num_time_steps_tr = 40000 #An orbit for saturn is 129110 steps
num_time_steps_sr = 500
num_batches = 400
#num_time_steps_val = int(total_time_traj/delta_time) - num_time_steps_tr


# Read the data
nplanets = 1 # Number of planets (not counting the sun)
data, masses, names = read_orbits.main(nplanets = nplanets, frame='b', use_moons = True, 
                                       path='/Users/Pablo/Dropbox/data/orbits/7parts/part1/')
nplanets = len(data[0])
nedges = nplanets*(nplanets-1)//2
batch_size_tr = num_time_steps_tr//num_batches

print('Formatting data')

# Get the acceleration
A = data[1:,:,3:] - data[:-1,:,3:]
data[:-1, :, 3:] = A/delta_time 
data = data[:-1]

# For debugging, reduce size of validation data. Really speeds things up!
nval = 10000
data = data[:(num_time_steps_tr + nval)]

masses/=MEARTH#/1000000

# Split into training and validation
data_tr = data[:num_time_steps_tr]
data_val = data[num_time_steps_tr:]

num_time_steps_val = len(data_val)

# Shuffle the data
np.random.shuffle(data_tr)
np.random.shuffle(data_val)

D_tr_np = np.empty([len(data_tr), nedges, 3])
D_val_np = np.empty([len(data_val), nedges, 3])
F_val = np.empty([len(data_val), nedges, 3])
k=0
names_edges = []
senders, receivers = [], []
for i in range(nplanets):
    for j in range(nplanets):
        if i > j:
            d_tr = data_tr[:,j,:3] - data_tr[:,i,:3]
            d_val = data_val[:,j,:3] - data_val[:,i,:3]
            D_tr_np[:,k,:] = d_tr
            D_val_np[:,k,:] = d_val 
            F_val[:,k,:] = force_newton(d_val, masses[i], masses[j]) 
            names_edges.append(names[j] + ' - ' + names[i])
            k+=1 
            receivers.append(i)
            senders.append(j)

A_tr = data_tr[:,:,3:]
A_val = data_val[:,:,3:]
A_norm =np.std(A_tr) 

D_tr_flat = np.reshape(D_tr_np, [num_time_steps_tr*nedges, 3])
D_val_flat = np.reshape(D_val_np,[1, num_time_steps_val*nedges, 3])

A_tr_flat = np.reshape(A_tr/A_norm, [num_time_steps_tr*nplanets, 3])
A_val_flat = np.reshape(A_val/A_norm, [1, num_time_steps_val*nplanets, 3])

D_tr = tf.convert_to_tensor(D_tr_flat, dtype="float32")
A_tr = tf.convert_to_tensor(A_tr_flat, dtype="float32")
D_tr_batches = tf.split(D_tr,  num_batches)
A_tr_batches = tf.split(A_tr,  num_batches)

D_val = tf.convert_to_tensor(D_val_flat, dtype="float32")
A_val = tf.convert_to_tensor(A_val_flat, dtype="float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (D_tr_batches, A_tr_batches))#.batch(batch_size_tr)

test_ds = tf.data.Dataset.from_tensor_slices(
    (D_val, A_val))

#masses_tf = tf.convert_to_tensor(masses, dtype="float32")

print('Cleaning memory')
A_tr = None
A_val = None
D_tr = None
D_val = None
A_tr_flat = None
A_val_flat = None
D_tr_flat = None
D_val_flat = None
A_tr_batches = None
D_tr_batches = None
data = None
data_tr = None
gc.collect()

checkpoint_filepath = './saved_models/sph_32_2'

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                            verbose = 1,
                                            patience=50,
                                            #baseline = 0.1,
                                            restore_best_weights=False)
# Restore best weights not working, but found way around using checkpoint

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                 save_weights_only=False,
                                                save_best_only=True,
                                                 verbose=0)
model = LearnForces(nplanets, senders, receivers, noise_level = noise_level)

#model.compile(run_eagerly=True)
model.compile()

print('Starting model training')
model.fit(train_ds, 
          epochs = 1000, 
          verbose=2,
          callbacks=[early_stopping, checkpoint], 
          validation_data=test_ds
         )

print('Model training completed')

model.load_weights(checkpoint_filepath)

#model.evaluate(test_ds)

print ('Learned planetary masses:')
print(np.round(model.logm_planets.numpy(), 2))

print ('True planetary masses:')
print(np.round(np.log10(masses)[1:], 2))

print('Generating plot')
ap ,fp = model(D_val_flat[0], extract = True)

nrows = math.ceil(nplanets/4)
fig, ax = plt.subplots(nrows, 4, figsize = (16, 4*nrows))
for i in range(nplanets):
    ax[i//4, i%4].set_title(names[i], fontsize = 16)
    ax[i//4, i%4].plot(ap[:,i,0], ap[:,i,1], '.', label = 'Learned')
    ax[i//4, i%4].plot(data_val[:,i,3]/A_norm, data_val[:,i,4]/A_norm, '.', label = 'Truth')

ax[0,0].legend()
plt.savefig('./learned_orbits.png')


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