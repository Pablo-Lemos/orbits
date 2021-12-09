# -*- coding: utf-8 -*-

"""
Plotter for learned vs true trajectories
"""
if __name__ == "__main__":
    ###############################################################################
    # initial imports:

    import numpy as np
    import os
    from simulate_orbits import *
    import read_orbits
    from matplotlib import pylab as plt
    import matplotlib.gridspec as gridspec
    from copy import copy

    ###############################################################################
    # initial setup:
    # output folder:
    out_folder = '../paper_plots/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    nplanets = 8  #  Number of planets (not counting the sun)
    masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                     path='/Users/pl332/Dropbox/data/orbits/7parts/full//',
                                     read_data=False)

    data = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_data.npy')
    data_gnets = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_gnets.npy')
    sim_learned = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_sr.npy')
    final = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_sr_masses.npy')

    learned_masses = np.load('../saved_models/learned_masses_1.npy')

    nplanets = len(data[0])
    nedges = nplanets * (nplanets - 1) // 2

    ###############################################################################
    # Generate plot

    # color palette:
    colors = ['gold', 'grey', 'brown', 'darkblue', 'lightblue', 'red', 'orange']

    # latex rendering:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 14
    main_fontsize = 10.0

    # Start plot
    fig = plt.gcf()
    fig.set_size_inches(x_size / 2.54, y_size / 2.54)
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 3])
    ax11 = plt.subplot(gs[0, 0])
    ax21 = plt.subplot(gs[1, 0])
    ax31 = plt.subplot(gs[2, 0])

    ax12 = plt.subplot(gs[0, 1])
    ax22 = plt.subplot(gs[1, 1])
    ax32 = plt.subplot(gs[2, 1])

    ax11.set_xlim(-2, 2)
    ax11.set_ylim(-2, 2)

    ax21.set_xlim(-2, 2)
    ax21.set_ylim(-2, 2)

    ax31.set_xlim(-2, 2)
    ax31.set_ylim(-2, 2)

    ax21.set_xticklabels([])
    ax11.set_xticklabels([])
    ax21.set_xticks([])
    ax11.set_xticks([])

    ax22.set_xticklabels([])
    ax12.set_xticklabels([])
    ax22.set_xticks([])
    ax12.set_xticks([])

    ax31.set_xlabel('X [AU]')
    ax32.set_xlabel('X [AU] + $\Delta t$')
    ax11.set_ylabel('Y [AU]')
    #ax2.set_xlabel('X [AU]')
    #ax3.set_xlabel('X [AU]')
    ax21.set_ylabel('Y [AU]')
    ax31.set_ylabel('Y [AU]')

    ax12.set_title('Graph Network', loc='left', fontsize = 1.2*main_fontsize)
    ax22.set_title('Graph Network + Symbolic Regression', loc='left', fontsize = 1.2*main_fontsize)
    ax32.set_title('Graph Network + Symbolic Regression + relearned masses', loc='left', fontsize = 1.2*main_fontsize)

    N1 = 87600 // 5 // 2
    N = 87600 // 5 * 7//4
    delta_t = 3/20000

    alphas = np.logspace(-3, 0, N1)
    for i in range(31):
        if i in [0, 3, 2]:
            ax11.plot(data[:N1, i, 0], data[:N1, i, 1], c=colors[i], lw = 3, alpha = 0.2, rasterized=True)
            ax21.plot(data[:N1, i, 0], data[:N1, i, 1], c=colors[i], lw = 3, alpha = 0.2, rasterized=True)
            ax31.plot(data[:N1, i, 0], data[:N1, i, 1], c=colors[i], lw = 3, alpha = 0.2, rasterized=True)
            ax11.plot(data_gnets[:N1, i, 0], data_gnets[:N1, i, 1], ls='--', c=colors[i],
                         rasterized=True)
            ax21.plot(sim_learned[:N1, i, 0], sim_learned[:N1, i, 1], ls='--', c=colors[i],
                         rasterized=True)
            ax31.plot(final[:N1, i, 0], final[:N1, i, 1], ls='--', c=colors[i], rasterized=True)
            #ax11.scatter(data_gnets[:N1, i, 0], data_gnets[:N1, i, 1], ls = '--', s=0.1, c=colors[i], alpha = alphas, rasterized=True)
            #ax21.scatter(sim_learned[:N1, i, 0], sim_learned[:N1, i, 1], ls = '--',s=0.1, alpha = alphas, c=colors[i], rasterized=True)
            #ax31.scatter(final[:N1, i, 0], final[:N1, i, 1], s=0.1, ls = '--', alpha = alphas, c=colors[i], rasterized=True)
            ax11.scatter(data[N1, i, 0], data[N1, i, 1], marker='o', alpha = 0.2, c=colors[i], rasterized=True)
            ax21.scatter(data[N1, i, 0], data[N1, i, 1], marker='o', alpha = 0.2, c=colors[i], rasterized=True)
            ax31.scatter(data[N1, i, 0], data[N1, i, 1], marker='o', alpha = 0.2, c=colors[i], rasterized=True)
            ax11.scatter(data_gnets[N1, i, 0], data_gnets[N1, i, 1], marker='o', c=colors[i], rasterized=True)
            ax21.scatter(sim_learned[N1, i, 0], sim_learned[N1, i, 1], marker='o', c=colors[i], rasterized=True)
            ax31.scatter(final[N1, i, 0], final[N1, i, 1], marker='o', c=colors[i], rasterized=True,
                        label=names[i].capitalize())

            ax12.plot(data[:N, i, 0]+np.arange(N)*delta_t, data[:N, i, 1], c=colors[i], alpha = 0.5, rasterized=True)
            ax12.plot(data_gnets[:N, i, 0]+np.arange(N)*delta_t, data_gnets[:N, i, 1], ls = '--', c=colors[i], rasterized=True)
            ax22.plot(data[:N, i, 0]+np.arange(N)*delta_t, data[:N, i, 1], c=colors[i], alpha = 0.5, rasterized=True)
            ax22.plot(sim_learned[:N, i, 0]+np.arange(N)*delta_t, sim_learned[:N, i, 1],  ls = '--', c=colors[i], rasterized=True)
            if i == 3:
                ax32.plot(data[:N, i, 0] + np.arange(N) * delta_t, data[:N, i, 1], c=colors[i], alpha=0.5 , label = 'Truth',
                         rasterized=True)
                ax32.plot(final[:N, i, 0]+np.arange(N)*delta_t, final[:N, i, 1],  ls = '--', c=colors[i], rasterized=True, label = 'Learned')
            else:
                ax32.plot(data[:N, i, 0] + np.arange(N) * delta_t, data[:N, i, 1], c=colors[i], alpha=0.5,
                         rasterized=True)
                ax32.plot(final[:N, i, 0]+np.arange(N)*delta_t, final[:N, i, 1],  ls = '--', c=colors[i], rasterized=True)

            ax12.scatter(data[N, i, 0]+N*delta_t, data[N, i, 1], marker='o', alpha = 0.5, c=colors[i], rasterized=True)
            ax22.scatter(data[N, i, 0] + N * delta_t, data[N, i, 1], marker='o', alpha = 0.5, c=colors[i], rasterized=True)
            ax32.scatter(data[N, i, 0] + N * delta_t, data[N, i, 1], marker='o', alpha = 0.5, c=colors[i], rasterized=True)
            ax12.scatter(data_gnets[N, i, 0]+N*delta_t, data_gnets[N, i, 1], marker='o', c=colors[i], rasterized=True)
            ax22.scatter(sim_learned[N, i, 0]+N*delta_t, sim_learned[N, i, 1], marker='o', c=colors[i], rasterized=True, label = names[i].capitalize())
            ax32.scatter(final[N, i, 0]+N*delta_t, final[N, i, 1], marker='o', c=colors[i], rasterized=True)

    # legends:
    leg = ax22.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     #loc='right',
                     bbox_to_anchor=(0.78, 0.56)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    leg = ax32.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     #loc='right',
                     bbox_to_anchor=(0.74, 0.64)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # update dimensions:
    bottom = 0.1
    top = 0.95
    left = 0.09
    right = 0.99
    wspace = 0.12
    hspace = .3
    gs.update(bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace)

    plt.savefig(out_folder + 'plot_rollout.pdf', dpi=200)
    plt.show()