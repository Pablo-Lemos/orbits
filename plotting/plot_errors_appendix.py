"""
Plotter for rollout errors
"""
if __name__ == "__main__":
    ###############################################################################
    # initial imports:

    import numpy as np
    import os
    from simulate_orbits import *
    from simulator import read_orbits
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from copy import copy
    from data.solar_system_names import *
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
    colors = {'sun':'gold', 'mercury':'grey', 'venus':'brown', 'earth':'darkblue',
              'moon':'grey', 'mars':'red', 'jupiter': 'orange', 'neptune': 'darkblue'}

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 22
    main_fontsize = 10.0

    # Start plot
    fig = plt.gcf()
    fig.set_size_inches( x_size/2.54, y_size/2.54 )
    gs = gridspec.GridSpec(4,2)
    ax11 = plt.subplot(gs[0,0])
    ax12 = plt.subplot(gs[0,1])
    ax21 = plt.subplot(gs[1,0])
    ax22 = plt.subplot(gs[1,1])
    ax31 = plt.subplot(gs[2,0])
    ax32 = plt.subplot(gs[2,1])
    ax41 = plt.subplot(gs[3,0])
    ax42 = plt.subplot(gs[3,1])
    axs = [ax11, ax12, ax21, ax22, ax31, ax32, ax41, ax42]

    ax11.set_ylabel(
        r"$(x_{\rm truth} - x_{\rm pred})^2 / x_{\rm truth}^2$",
        fontsize=0.9 * main_fontsize);

    ax21.set_ylabel(
        r"$(x_{\rm truth} - x_{\rm pred})^2 / x_{\rm truth}^2$",
        fontsize=0.9 * main_fontsize);

    ax31.set_ylabel(
        r"$(x_{\rm truth} - x_{\rm pred})^2 / x_{\rm truth}^2$",
        fontsize=0.9 * main_fontsize);

    ax41.set_ylabel(
        r"$(x_{\rm truth} - x_{\rm pred})^2 / x_{\rm truth}^2$",
        fontsize=0.9 * main_fontsize);


    ax41.set_xlabel("Time (years)", fontsize=main_fontsize);
    ax42.set_xlabel("Time (years)", fontsize=main_fontsize);

    for ax in axs:
        ax.set_ylim(1e-13, 1e2)
        ax.set_yscale("log")

    ax11.set_title('Inner solar system', loc='center', fontsize=1 *
                                                               main_fontsize)
    ax12.set_title('Earth + moon', loc='center', fontsize=1 * main_fontsize)
    ax21.set_title('Outer solar system', loc='center', fontsize=1 *
                                                               main_fontsize)
    ax22.set_title('Jupiter + moons', loc='center', fontsize=1 *
                                                               main_fontsize)
    ax31.set_title('Saturn + moons 1', loc='center', fontsize=1 *
                                                               main_fontsize)
    ax32.set_title('Saturn + moons 2', loc='center', fontsize=1 *
                                                               main_fontsize)
    ax41.set_title('Uranus + moons', loc='center', fontsize=1 *
                                                               main_fontsize)
    ax42.set_title('Neptune + moons', loc='center', fontsize=1 *
                                                               main_fontsize)

    t = np.arange(N-1)/2/24/365.25
    k = 0
    for i in range(31):
        if names[i] in ['sun', 'mercury', 'earth', 'venus']:
            if names[i] in colors:
                ax11.plot(t, loss_gnets[1:, i], color =colors[names[i]],
                        label=names[i].capitalize())
            else:
                ax11.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in ['earth', 'moon']:
            if names[i] in colors:
                ax12.plot(t, loss_gnets[1:, i], color =colors[names[i]],
                        label=names[i].capitalize())
            else:
                ax12.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in ['jupiter', 'saturn', 'uranus', 'neptune']:
            if names[i] in colors:
                ax21.plot(t, loss_gnets[1:, i], color=colors[names[i]],
                          label=names[i].capitalize())
            else:
                ax21.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in ['jupiter'] + jupiter_moon_names:
            if names[i] in colors:
                ax22.plot(t, loss_gnets[1:, i], color=colors[names[i]],
                          label=names[i].capitalize())
            else:
                ax22.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in ['saturn'] + saturn_moon_names[:4]:
            if names[i] in colors:
                ax31.plot(t, loss_gnets[1:, i], color=colors[names[i]],
                          label=names[i].capitalize())
            else:
                ax31.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in saturn_moon_names[4:]:
            if names[i] in colors:
                ax32.plot(t, loss_gnets[1:, i], color=colors[names[i]],
                          label=names[i].capitalize())
            else:
                ax32.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in ['uranus'] + uranus_moon_names:
            if names[i] in colors:
                ax41.plot(t, loss_gnets[1:, i], color=colors[names[i]],
                          label=names[i].capitalize())
            else:
                ax41.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

        if names[i] in ['neptune'] + neptune_moon_names:
            if names[i] in colors:
                ax42.plot(t, loss_gnets[1:, i], color=colors[names[i]],
                          label=names[i].capitalize())
            else:
                ax42.plot(t, loss_gnets[1:, i], label=names[i].capitalize())

            k += 1

    # update dimensions:
    bottom=0.08; top=0.95; left=0.12; right=0.95; wspace=0.1; hspace=0.15
    gs.update( bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace )

    ax11.set_xticklabels([])
    ax12.set_xticklabels([])
    ax21.set_xticklabels([])
    ax22.set_xticklabels([])
    ax31.set_xticklabels([])
    ax32.set_xticklabels([])

    ax12.set_yticklabels([])
    ax22.set_yticklabels([])
    ax32.set_yticklabels([])
    ax42.set_yticklabels([])

    for ax in axs:
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
    plt.savefig(out_folder + 'plot_errors_appendix.pdf')

    plt.show()