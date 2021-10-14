# -*- coding: utf-8 -*-

"""
Plotter for learned vs true trajectories
"""
if __name__ == "__main__":
    ###############################################################################
    # initial imports:

    import numpy as np
    from simulate_orbits import *
    import read_orbits
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from copy import copy

    ###############################################################################
    # initial setup:
    nplanets = 8  #  Number of planets (not counting the sun)
    data, masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                           path='/Users/pablo/Dropbox/data/orbits/7parts/full//')
    nplanets = len(data[0])
    nedges = nplanets * (nplanets - 1) // 2

    learned_masses = np.load('../saved_models/learned_masses_7.npy')

    ###############################################################################
    # integrate the trajectories:
    bodies_sim = []
    bodies_learned = []
    for i in range(nplanets):
        body = Body()
        body.name = names[i]
        body.mass = masses[i] / masses[0]  # Solar masses
        body.pos = copy(data[0, i, :3])
        body.vel = copy(data[0, i, 3:])  # *365.25 # Convert velocity to AU/Y
        bodies_sim.append(body)

        body_learned = Body()
        body_learned.name = names[i]
        body_learned.mass = 10 ** (learned_masses[i])  # Solar masses
        body_learned.pos = copy(data[0, i, :3])
        body_learned.vel = copy(data[0, i, 3:])  # *365.25 # Convert velocity to AU/Y
        bodies_learned.append(body_learned)

    delta_time = 0.5 * (1 / 24.)  # *DAY/YEAR # 30 minutes
    total_time = 5. * 365  # 0.1 years
    # The G learned divided by the learned mass of the sun,
    G_learned = 19422207.0337081 / 10 ** (3.3537278) * 3.7630259666518835e-06 * 0.008888999709186746

    # sim_orbits = simulate(bodies_sim, total_time, delta_time, G)
    sim_learned = simulate(bodies_learned, total_time, delta_time, G_learned)

    ###############################################################################
    # Generate plot

    # color palette:
    colors = ['gold', 'grey', 'lightgrey', 'darkblue', 'lightblue', 'red']

    # latex rendering:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 6
    main_fontsize = 10.0

    # Start plot
    fig = plt.gcf()
    fig.set_size_inches(x_size / 2.54, y_size / 2.54)
    gs = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)

    ax2.set_yticklabels([])
    ax3.set_yticklabels([])

    ax1.set_xlabel('X [AU]')
    ax1.set_ylabel('Y [AU]')
    ax2.set_xlabel('X [AU]')
    ax3.set_xlabel('X [AU]')

    ax1.set_title('Data')
    ax2.set_title('Graph Nets')
    ax3.set_title('Symbolic Regression')

    N = 10000

    for i in range(6):
        ax1.scatter(data[:N, i, 0], data[:N, i, 1], s=0.1, c=colors[i], rasterized=True)
        ax2.scatter(data_gnets[:N, i, 0], data_gnets[:N, i, 1], s=0.1, c=colors[i], rasterized=True)
        ax3.scatter(sim_learned[:N, i, 0], sim_learned[:N, i, 1], s=0.1, c=colors[i], rasterized=True)
        ax1.scatter(data[N, i, 0], data[N, i, 1], marker='o', c=colors[i], rasterized=True)
        ax2.scatter(data_gnets[N, i, 0], data_gnets[N, i, 1], marker='o', c=colors[i], rasterized=True)
        ax3.scatter(sim_learned[N, i, 0], sim_learned[N, i, 1], marker='o', c=colors[i], rasterized=True)

    ax1.scatter(data[:N, 3, 0], data[:N, 3, 1], s=0.1, c=colors[3], rasterized=True)
    ax2.scatter(data_gnets[:N, 3, 0], data_gnets[:N, 3, 1], s=0.1, c=colors[3], rasterized=True)
    ax3.scatter(sim_learned[:N, 3, 0], sim_learned[:N, 3, 1], s=0.1, c=colors[3], rasterized=True)
    ax1.scatter(data[N, 3, 0], data[N, 3, 1], marker='o', c=colors[3], rasterized=True)
    ax2.scatter(data_gnets[N, 3, 0], data_gnets[N, 3, 1], marker='o', c=colors[3], rasterized=True)
    ax3.scatter(sim_learned[N, 3, 0], sim_learned[N, 3, 1], marker='o', c=colors[3], rasterized=True)

    # update dimensions:
    bottom = 0.25
    # top = 0.99
    # left = 0.01
    # right = 0.99
    # wspace = 0.
    # hspace = 0.
    gs.update(bottom=bottom,  # top=top, left=left, right=right,
              #                  wspace=wspace, hspace=hspace
              )

    plt.savefig('../paper_plots/plot_rollout.pdf')
