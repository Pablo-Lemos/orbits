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


    N = 6
    masses_learned = np.zeros([N, 31])
    masses_learned_fixed = np.zeros([N, 31])
    for i in range(N):
        masses_learned[i] = np.load('../saved_models/learned_masses_' + str(i + 1) + '.npy')
        masses_learned_fixed[i] = np.load('../saved_models/learned_masses_fixed_' + str(i + 1) + '.npy')

    nplanets = 8  #  Number of planets (not counting the sun)
    masses, names = read_orbits.main(nplanets=nplanets, frame='b', use_moons=True,
                                           path='/Users/Pablo/Dropbox/data/orbits/7parts/part1/',
                                           read_data=False)

    logmasses = np.log10(masses/masses[0])
    mass_error = np.mean((masses_learned - logmasses.reshape([1,-1]))**2, axis = 0)#/logmasses**2
    mass_error_fixed = np.mean((masses_learned_fixed - logmasses.reshape([1,-1]))**2, axis = 0)#/logmasses**2

    indices_moons = [*range(1,31)]
    indices_planets = []
    for i, planet in enumerate(planet_names):
        #print(indices_moons, names.index(planet), planet)
        newplanet = names.index(planet)
        indices_moons.remove(newplanet)
        indices_planets.append(newplanet)

    nplanets = len(indices_planets)
    nmoons = len(indices_moons)

    summed_forces = np.load('../saved_models/summed_forces.npy')
    summed_potentials = np.load('../saved_models/summed_potentials.npy')


    for i, name in enumerate(names):
        names[i] = name.capitalize()

    ###############################################################################
    # do the plot:

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 12.0
    main_fontsize = 10.0

    # start the plot:
    fig = plt.gcf()
    fig.set_size_inches( x_size/2.54, y_size/2.54 )
    gs = gridspec.GridSpec(2,2, width_ratios=[2, 1])
    ax11 = plt.subplot(gs[0,0])
    ax12 = plt.subplot(gs[0,1])
    ax21 = plt.subplot(gs[1,0])
    ax22 = plt.subplot(gs[1,1])

    # do the plot:
    ax11.plot(np.arange(nplanets), np.log10(masses[indices_planets] / masses[0]), "*", label='Truth', color = colors[2])
    ax21.plot(np.arange(nplanets), np.log10(masses[indices_planets] / masses[0]), "*", label='Truth', color = colors[2])
    for i in range(N-1):
        ax11.plot(np.arange(nplanets), masses_learned[i, indices_planets], "o", color = 'darkred', alpha = 0.2, markersize=3)
        ax21.plot(np.arange(nplanets), masses_learned_fixed[i, indices_planets], "o", color='darkgreen', alpha=0.2,
                  markersize=3)
    ax11.plot(np.arange(nplanets), masses_learned[N-1, indices_planets], "o", color='darkred', alpha=0.2, markersize=3, label='Predicted')
    ax21.plot(np.arange(nplanets), masses_learned_fixed[N - 1, indices_planets], "o", color='darkgreen', alpha=0.2,
              markersize=3, label='Predicted')

    ax11.plot(nplanets + 1 + np.arange(nmoons), np.log10(masses[indices_moons] / masses[0]), "*", color = colors[2])
    ax21.plot(nplanets + 1 + np.arange(nmoons), np.log10(masses[indices_moons] / masses[0]), "*", color=colors[2])
    for i in range(N):
        ax11.plot(nplanets + 1 + np.arange(nmoons), masses_learned[i, indices_moons], "o", color = 'salmon', alpha = 0.2, markersize=3)
        ax21.plot(nplanets + 1 + np.arange(nmoons), masses_learned_fixed[i, indices_moons], "o", color='lightgreen', alpha=0.2,
                  markersize=3)

    moon_mapping = map(names.__getitem__, indices_moons)
    moon_names = list(moon_mapping)

    ax11.axvline(nplanets, ls = '--', color ='k')
    ax21.axvline(nplanets, ls = '--', color ='k')

    xticks = np.arange(30)
    xticks[nplanets:] = xticks[nplanets:] + 1
    ax21.set_xticks(xticks)
    ax21.set_xticklabels(planet_names + moon_names, rotation=60, ha='right' , size='small');
    ax21.set_xlim([-0.5, 30.5])

    ax11.set_xticks(np.arange(30))
    #ax11.set_xticklabels(planet_names + moon_names, rotation=60, ha='right' , size='small');
    ax11.set_xticklabels([])
    ax11.set_xlim([-0.5, 30.5])
    # label on the axis:
    ax11.set_ylabel(r'$\log_{10} \left( M / M_{\odot} \right)$' , fontsize=main_fontsize);
    ax21.set_ylabel(r'$\log_{10} \left( M / M_{\odot} \right)$' , fontsize=main_fontsize);

    # Hide the right and top spines
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax21.spines['right'].set_visible(False)
    ax21.spines['top'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.spines['top'].set_visible(False)
    ax22.spines['right'].set_visible(False)
    ax22.spines['top'].set_visible(False)

    ax12.plot(summed_potentials[indices_planets], mass_error[indices_planets], "o", color='darkred', alpha=1,
             markersize=3, label="Planets")
    ax12.plot(summed_potentials[indices_moons], mass_error[indices_moons], "o", color='salmon', alpha=1,
             markersize=3, label="Moons")
    ax22.plot(summed_potentials[indices_planets], mass_error_fixed[indices_planets], "o", color='darkgreen', alpha=1,
             markersize=3, label="Planets")
    ax22.plot(summed_potentials[indices_moons], mass_error_fixed[indices_moons], "o", color='lightgreen', alpha=1,
             markersize=3, label="Moons")

    ax12.annotate("Hyperion", (summed_potentials[18], mass_error[18]), fontsize=0.7 * main_fontsize)
    ax12.annotate("Phoebe", (summed_potentials[20], mass_error[20]+9), fontsize=0.7 * main_fontsize)
    ax12.annotate("Nereid", (summed_potentials[29], mass_error[29]-7), fontsize=0.7 * main_fontsize)
    ax22.annotate("Hyperion", (summed_potentials[18], mass_error_fixed[18]), fontsize=0.7 * main_fontsize)
    ax22.annotate("Phoebe", (summed_potentials[20], mass_error_fixed[20]), fontsize=0.7 * main_fontsize)
    ax22.annotate("Nereid", (summed_potentials[29], mass_error_fixed[29]-0.1), fontsize=0.7 * main_fontsize)

    ax22.set_xscale('log')
    ax22.set_yscale('log')

    ax12.set_xscale('log')
    ax12.set_yscale('log')

    ax11.set_title('Fig. 4A: Graph network + symbolic regression', loc='center',
                   fontsize=1.1 * main_fontsize)
    ax12.set_title('Fig. 4B: Grav influence',
                   loc='center',
                   fontsize=1.1 * main_fontsize)
    ax21.set_title('Fig. 4C: Graph network + symbolic regression + relearned '
                   'masses', loc='center', fontsize=1.1 * main_fontsize)
    ax22.set_title('Fig. 4D: Grav influence',
                   loc='center',
                   fontsize=1.1 * main_fontsize)


    ax12.set_xticklabels([])
    ax12.set_ylim(1e-8, 1e2)
    ax22.set_ylim(1e-8, 1e2)

    # update dimensions:
    bottom=0.15; top=0.92; left=0.08; right=0.99; wspace=0.24; hspace=0.2
    gs.update( bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace )

    #ax12.set_xlabel("Gravitational influence $\mathrm{[AU^{-2} \ yr^{-2}]}$", fontsize=0.9*main_fontsize);
    ax12.set_ylabel(r"Error in $\log_{10} \left( M/M_{\odot} \right)$", fontsize=0.9*main_fontsize);
    ax22.set_xlabel("Gravitational influence $\mathrm{[AU^{-2} \ yr^{-2}]}$", fontsize=0.9*main_fontsize);
    ax22.set_ylabel(r"Error in $\log_{10} \left( M/M_{\odot} \right)$", fontsize=0.9*main_fontsize);

    # legends:
    leg = ax11.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='upper left',
                     bbox_to_anchor=(0.74, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    leg = ax21.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='upper left',
                     bbox_to_anchor=(0.74, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # legends:
    leg = ax12.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='upper right',
                     #bbox_to_anchor=(0.4, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # legends:
    leg = ax22.legend(fontsize=0.9 * main_fontsize,
                     frameon=True,
                     fancybox=True,
                     edgecolor='k',
                     ncol=1,
                     borderaxespad=0.0,
                     columnspacing=2.0,
                     handlelength=1.4,
                     loc='upper right',
                     #bbox_to_anchor=(0.4, 0.96)
                     )
    leg.get_frame().set_linewidth('0.8')
    leg.get_title().set_fontsize(main_fontsize)

    # save:
    plt.savefig(out_folder+'plot_masses.pdf')

    plt.show()