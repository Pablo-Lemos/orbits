# -*- coding: utf-8 -*-

"""
Plotter for symbolic regression plot
"""
if __name__ == "__main__":
    ###############################################################################
    # initial imports:

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from plotting import color_utilities_orig as cu
    import matplotlib.gridspec as gridspec
    from simulator import read_orbits
    from data_symreg import *

    ###############################################################################
    # initial setup:
    def loss(y_pred, y_true):
        x = (y_true - y_pred)
        x = np.sum(x ** 2, axis=-1)
        # x = np.sum(x, axis = 0)

        x2 = y_true
        x2 = np.sum(x2 ** 2, axis=-1)
        # x2 = np.sum(x2, axis = 0)

        return x/x2 #np.sum(x / x2, axis=-1)

    # output folder:
    out_folder = '../paper_plots/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # color palette:
    colors = [cu.nice_colors(i) for i in range(10)]
    colors[3] = 'pink'

    # latex rendering:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    ###############################################################################
    # pre computations:

    centroids_all = []
    scores_plot = []
    eqs_plot = []
    xticks = []
    xticklabels = []

    ii = 0
    for i, eq in enumerate(eq_list):
        scores = scores_list[i]
        if len(scores) > 2:
            centroids = ii + np.arange(len(scores))
            centroids_all.append(centroids)
            ii += len(scores)+1
            scores_plot.append(scores)
            eqs_plot.append(eq)
            xticks.append(np.mean(centroids))
            xticklabels.append(comp_list[i])

    nplanets = 8  #  Number of planets (not counting the sun)
    masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                     path='/Users/pl332/Dropbox/data/orbits/7parts/full//',
                                     read_data=False)


    PATH = "../saved_models"
    data = np.load(PATH + '/arrays/array_data.npy')
    sim_learned = np.load(PATH + '/arrays/array_sr.npy')
    sim_learned_masses = np.load(PATH + '/arrays/array_sr_masses.npy')
    newtonian = np.load(PATH + '/arrays/newtonian.npy')

    sim_learned_eq1 = np.load(PATH + '/array_eqns/eq1.npy')
    sim_learned_eq2 = np.load(PATH + '/array_eqns/eq2.npy')
    sim_learned_eq3 = np.load(PATH + '/array_eqns/eq3.npy')
    sim_learned_eq4 = np.load(PATH + '/array_eqns/eq4.npy')
    sim_learned_eq5 = np.load(PATH + '/array_eqns/eq5.npy')
    sim_learned_eq5b = np.load(PATH + '/array_eqns/eq5b.npy')

    N = 87600 // 5 * 3
    loss_learned = loss(sim_learned[:N, :, :3], data[:N, :, :3])
    loss_learned_masses = loss(sim_learned_masses[:N, :, :3], data[:N, :, :3])

    loss_eq1 = loss(sim_learned_eq1[:N, :, :3], data[:N, :, :3])
    loss_eq2 = loss(sim_learned_eq2[:N, :, :3], data[:N, :, :3])
    loss_eq3 = loss(sim_learned_eq3[:N, :, :3], data[:N, :, :3])
    loss_eq4 = loss(sim_learned_eq4[:N, :, :3], data[:N, :, :3])
    loss_eq5 = loss(sim_learned_eq5[:N, :, :3], data[:N, :, :3])
    loss_eq5b = loss(sim_learned_eq5b[:N, :, :3], data[:N, :, :3])
    loss_newtonian = loss(newtonian[:N, :, :3], data[:N, :, :3])

    losses_eqns = [loss_eq1, loss_eq2, loss_eq3, loss_eq4, loss_eq5, loss_learned]

    ###############################################################################
    # do the plot:

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 8.0
    main_fontsize = 10.0

    # start the plot:
    fig = plt.gcf()
    fig.set_size_inches( x_size/2.54, y_size/2.54 )
    gs = gridspec.GridSpec(1,2, width_ratios=(1, 1))
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])

    # do the plot:
    for i, score in enumerate(scores_plot):
        #ax1.hist(centroids_all[i], bins=len(score), weights=score, color=colors[i], edgecolor='black',
        #         label=eqs_plot[i])
        ax1.hist(i*5, bins = 1, weights = [max(score)], color = colors[i], edgecolor = 'black', label = eqs_plot[i])

    #ax1.set_xticks(xticks);
    ax1.set_xticks(5*np.arange(len(scores_plot)))
    #ax1.set_xticklabels(eqs_plot);
    ax1.set_xticklabels(xticklabels);

    # label on the axis:
    ax1.set_xlabel('Complexity ', fontsize=main_fontsize);
    ax1.set_ylabel(r'Score = $\delta$ Accuracy / $\delta$ Complexity', fontsize=0.9*main_fontsize);

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    #ax1.spines['bottom'].set_alpha(0.3)

    ax2.set_yscale("log")

    t = np.arange(N-1)/2/24/365.25
    for i in range(6):
        if i == 5:
            ax2.plot(t, np.sum(losses_eqns[i][1:, :], axis = 1), color =colors[i], label = 'NN + SR')
        else:
            ax2.plot(t, np.sum(losses_eqns[i][1:, :], axis=1), color=colors[i])

    ax2.plot(t, np.sum(loss_eq5b[1:, :], axis = 1), color =colors[4], ls = '--')
    ax2.plot(t, np.sum(loss_learned_masses[1:, :], axis = 1), color =colors[5], ls = '--', label = 'NN+ SR + relearned masses')
    ax2.plot(t, np.sum(loss_newtonian[1:, :], axis = 1), color ='k', lw = 5, alpha = 0.2, label = 'Newtonian gravity')

    ax2.set_xlabel("Time (years)", fontsize=main_fontsize);
    ax2.set_ylabel(r"Error = $(x_{\rm truth} - x_{\rm pred})^2 / x_{\rm truth}^2$", fontsize=0.9*main_fontsize);
    ax2.set_ylim([1e-7, 5e9])
    # legends:
    leg = ax2.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='lower right',
                     #bbox_to_anchor=(0.4, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # legends:
    leg = ax1.legend(fontsize=0.9 * main_fontsize,
                     frameon=False,
                     fancybox=False,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='upper left',
                     bbox_to_anchor=(0.04, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)


    # update dimensions:
    bottom=0.15; top=0.99; left=0.07; right=0.99; wspace=0.2; hspace=0.05
    gs.update( bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace )

    # save:
    plt.savefig(out_folder+'plot_symreg.pdf')

    plt.show()