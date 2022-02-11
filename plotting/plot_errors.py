"""
Plotter for rollout errors
"""
if __name__ == "__main__":
    ###############################################################################
    # initial imports:

    import numpy as np
    import os
    from simulate_orbits import *
    import read_orbits
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from copy import copy
    from solar_system_names import *
    import matplotlib.style as style

    style.use('tableau-colorblind10')


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

    nplanets = 8  #  Number of planets (not counting the sun)
    masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                     path='/Users/pl332/Dropbox/data/orbits/7parts/full//',
                                     read_data=False)

    data = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_data.npy')
    data_gnets = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_gnets.npy')
    sim_learned = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_sr.npy')
    final = np.load('/Users/pl332/Documents/PycharmProjects/orbits/saved_models/arrays/array_sr_masses.npy')

    learned_masses = np.load('../saved_models/learned_masses_7.npy')

    nplanets = len(data[0])
    nedges = nplanets * (nplanets - 1) // 2

    N = 87600 // 5 * 3
    loss_gnets = loss(data_gnets[:N, :, :3], data[:N, :, :3])
    loss_sr = loss(sim_learned[:N, :, :3], data[:N, :, :3])
    loss_sr2 = loss(final[:N, :, :3], data[:N, :, :3])

    ###############################################################################
    # Generate plot

    # color palette:
    colors = ['gold', 'grey', 'brown', 'darkblue', 'grey', 'red',
     'orange']

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 8.99
    y_size = 6
    main_fontsize = 10.0

    # Start plot
    fig = plt.gcf()
    fig.set_size_inches( x_size/2.54, y_size/2.54 )
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0,0])
    ax.set_ylim(1e-13, 1e2)

    ax.set_yscale("log")
    ax.set_title('Graph Network', loc='center', fontsize=1.2*main_fontsize)

    t = np.arange(N-1)/2/24/365.25
    for i in range(31):
        if i in [0, 2, 3]:
            ax.plot(t, loss_gnets[1:, i], color =colors[i], label=names[
                i].capitalize())

    ax.set_xlabel("Time (years)", fontsize=main_fontsize);
    #ax.set_ylabel(r"Error = $({\rm truth} - {\rm pred})^2 / {\rm truth}^2$", fontsize=0.9*main_fontsize);
    ax.set_ylabel(r"Error = $(x_{\rm truth} - x_{\rm pred})^2 / x_{\rm truth}^2$", fontsize=0.9 * main_fontsize);

    # update dimensions:
    bottom=0.2; top=0.9; left=0.25; right=0.95; wspace=0.03; hspace=0.05
    gs.update( bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace )

    # legends:
    leg = ax.legend(fontsize=0.9 * main_fontsize,
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
    plt.savefig(out_folder + 'plot_errors.pdf')

    plt.show()