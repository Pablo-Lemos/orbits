# -*- coding: utf-8 -*-

"""
Plotter for learned vs true masses
"""
if __name__ == "__main__":
    ###############################################################################
    # initial imports:

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import color_utilities_orig as cu
    import matplotlib.gridspec as gridspec
    import read_orbits
    from solar_system_names import *

    ###############################################################################
    # initial setup:

    # output folder:
    out_folder = '../paper_plots/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # color palette:
    colors = [cu.nice_colors(i) for i in range(4)]

    # latex rendering:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    ###############################################################################
    # pre computations:

    N = 9
    masses_learned = np.zeros([N, 31])
    for i in range(N):
        masses_learned[i] = np.load('../saved_models/learned_masses_fixed_' + str(i + 1) + '.npy')

    nplanets = 8  #  Number of planets (not counting the sun)
    masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                           path='/Users/Pablo/Dropbox/data/orbits/7parts/part1/',
                                           read_data=False)

    indices_moons = [*range(1,31)]
    indices_planets = []
    for i, planet in enumerate(planet_names):
        #print(indices_moons, names.index(planet), planet)
        newplanet = names.index(planet)
        indices_moons.remove(newplanet)
        indices_planets.append(newplanet)

    nplanets = len(indices_planets)
    nmoons = len(indices_moons)

    ###############################################################################
    # do the plot:

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 6.0
    main_fontsize = 10.0

    # start the plot:
    fig = plt.gcf()
    fig.set_size_inches( x_size/2.54, y_size/2.54 )
    gs = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0,0])

    # do the plot:
    ax1.plot(np.arange(nplanets), np.log10(masses[indices_planets] / masses[0]), "*", label='Truth', color = colors[2])
    for i in range(N-1):
        ax1.plot(np.arange(nplanets), masses_learned[i, indices_planets], "o", color = colors[0], alpha = 0.2, markersize=3)
    ax1.plot(np.arange(nplanets), masses_learned[N-1, indices_planets], "o", color=colors[0], alpha=0.2, markersize=3, label='Predicted')

    ax1.plot(nplanets + np.arange(nmoons), np.log10(masses[indices_moons] / masses[0]), "*", color = colors[2])
    for i in range(N):
        ax1.plot(nplanets + np.arange(nmoons), masses_learned[i, indices_moons], "o", color = colors[0], alpha = 0.2, markersize=3)

    moon_mapping = map(names.__getitem__, indices_moons)
    moon_names = list(moon_mapping)

    ax1.axvline(nplanets-0.5, ls = '--', color ='k')

    ax1.set_xticks(np.arange(30))
    ax1.set_xticklabels(planet_names + moon_names, rotation=60, ha='right' , size='small');
    ax1.set_xlim([-1, 33])
    # label on the axis:
    ax1.set_ylabel(r'$\log_{10} \left( M / M_{\odot} \right)$' , fontsize=main_fontsize);

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # update dimensions:
    bottom=0.26; top=0.99; left=0.10; right=0.99; wspace=0.03; hspace=0.05
    gs.update( bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace )

    # legends:
    leg = ax1.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='upper left',
                     bbox_to_anchor=(0.84, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # save:
    plt.savefig(out_folder+'plot_masses_better.pdf')

    plt.show()