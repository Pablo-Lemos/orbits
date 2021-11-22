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
    masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                     path='/Users/pl332/Dropbox/data/orbits/7parts/full//',
                                     read_data=False)

    data = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/array_data.npy')
    data_gnets = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/array_gnets.npy')
    sim_learned = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/array_sr.npy')
    learned_masses = np.load('../saved_models/learned_masses_7.npy')

    nplanets = len(data[0])
    nedges = nplanets * (nplanets - 1) // 2

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
    ax2.set_title('Neural Network')
    ax3.set_title('Symbolic Regression')

    N = 10000

    for i in range(6):
        ax1.scatter(data[:N, i, 0], data[:N, i, 1], s=0.1, c=colors[i], rasterized=True)
        ax2.scatter(data_gnets[:N, i, 0], data_gnets[:N, i, 1], s=0.1, c=colors[i], rasterized=True)
        ax3.scatter(sim_learned[:N, i, 0], sim_learned[:N, i, 1], s=0.1, c=colors[i], rasterized=True)
        ax1.scatter(data[N, i, 0], data[N, i, 1], marker='o', c=colors[i], rasterized=True)
        ax2.scatter(data_gnets[N, i, 0], data_gnets[N, i, 1], marker='o', c=colors[i], rasterized=True)
        ax3.scatter(sim_learned[N, i, 0], sim_learned[N, i, 1], marker='o', c=colors[i], rasterized=True, label = names[i].capitalize())

    ax1.scatter(data[:N, 3, 0], data[:N, 3, 1], s=0.1, c=colors[3], rasterized=True)
    ax2.scatter(data_gnets[:N, 3, 0], data_gnets[:N, 3, 1], s=0.1, c=colors[3], rasterized=True)
    ax3.scatter(sim_learned[:N, 3, 0], sim_learned[:N, 3, 1], s=0.1, c=colors[3], rasterized=True)
    ax1.scatter(data[N, 3, 0], data[N, 3, 1], marker='o', c=colors[3], rasterized=True)
    ax2.scatter(data_gnets[N, 3, 0], data_gnets[N, 3, 1], marker='o', c=colors[3], rasterized=True)
    ax3.scatter(sim_learned[N, 3, 0], sim_learned[N, 3, 1], marker='o', c=colors[3], rasterized=True)

    # legends:
    leg = ax3.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     #loc='right',
                     bbox_to_anchor=(1.05, 0.9)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # update dimensions:
    bottom = 0.2
    top = 0.85
    left = 0.1
    right = 0.85
    wspace = 0.
    hspace = 0.
    gs.update(bottom=bottom, top=top, left=left, right=right,# wspace=wspace, hspace=hspace
              )

    #plt.show()
    plt.savefig('../paper_plots/plot_rollout.pdf', dpi=200)
