#!/usr/bin/env python3

'''
Performs leapfrog integration to recover orbits from a model. 

Created by Pablo Lemos (UCL) 
28-11-2019
pablo.lemos.18@ucl.ac.uk
'''
import numpy as np

def get_leapfrog_step(x, ph, dt, m1, m2, model):
    ''' 
    Calculate the next step for leapfrog integration

    Parameters
    ----------
    x: float(4)
        Position at time t, in format (x1, y1, x2, y2)
    ph: float(2)
        Momentum at time (t+dt/2), in format (px, py)
    dt: float
        The time step size
    m1, m2: float 
        mass of each body
    model: MLP
        The model used to predict the momentum update
    norm_p: float
        The normalization of the momentum used by the MLP. Defaults to 1
            
    Returns
    -------
    x1: float(4)
        Position at time t+dt
    ph3: float(2)
        Momentum at time (t+dt*3/2)
    dp: float(2)
        Momentum update
    '''
    # x_{n+1}
    x1 = np.zeros(4)
    x1[:2] = x[:2] + dt*ph/m1
    x1[2:] = x[2:] + dt*ph/m2
    
    # Distance
    dX = x1[2:] - x1[:2]

    # p_{n+3/2}d
    dp = model(dX)
    ph3 = ph + dp

    return x1, ph3, dp


def integrate_leapfrog(X0, P0, dt, m1, m2, nsteps, model):
    '''
    Leapfrog integration, following notation from 
    https://young.physics.ucsc.edu/115/leapfrog.pdf
    
    Parameters
    ----------
    X0: float(4)
        Initial Position, in format (x1, y1, x2, y2)
    P0: float(2)
        Initial momentum, in format (p1, p2)
    dt: float
        The time step size
    m1, m2: float 
        mass of each body
    nsteps: int
        Number of integration steps
    model: MLP
        The model used to predict the momentum update
    norm_p: float
        The normalization of the momentum used by the MLP. Defaults to 1

    Returns
    -------
    x_pred, p_pred
        The predicted position and momenta
    '''

    # Distance
    dX = X0[2:] - X0[:2]

    # Model predict*norm_p = F*dt
    # (ph = phalf)
    ph = P0 + 0.5*model(dX)
    x = np.copy(X0)
    
    x_pred = np.zeros([nsteps, 4])
    dp_pred = np.zeros([nsteps-1, 2])
    
    if X0.ndim == 1:
        x_pred[0] = X0
    else: 
        x_pred = X0[:,0]
    for i in range(nsteps-1):
        x, ph, dp = get_leapfrog_step(x, ph, dt, m1, m2, model)
        x_pred[i+1] = x
        dp_pred[i] = dp
        
    return x_pred, dp_pred
