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
    import color_utilities_orig as cu
    import matplotlib.gridspec as gridspec
    from data_symreg import *

    ###############################################################################
    # initial setup:

    # output folder:
    out_folder = '../paper_plots/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # color palette:
    colors = [cu.nice_colors(i) for i in range(10)]

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

    ###############################################################################
    # do the plot:

    # plot size in cm. Has to match to draft to make sure font sizes are consistent
    x_size = 18
    y_size = 7.0
    main_fontsize = 10.0

    # start the plot:
    fig = plt.gcf()
    fig.set_size_inches( x_size/2.54, y_size/2.54 )
    gs = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0,0])

    # do the plot:
    for i, score in enumerate(scores_plot):
        ax1.hist(centroids_all[i], bins = len(score), weights = score, color = colors[i], label = eqs_plot[i])

    ax1.set_xticks(xticks);
    ax1.set_xticklabels(xticklabels);

    # label on the axis:
    ax1.set_xlabel('Complexity', fontsize=main_fontsize);
    ax1.set_ylabel('Score', fontsize=main_fontsize);

    # update dimensions:
    bottom=0.15; top=0.99; left=0.09; right=0.99; wspace=0.03; hspace=0.05
    gs.update( bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace )

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

    # save:
    plt.savefig(out_folder+'plot_symreg.pdf')

    plt.show()