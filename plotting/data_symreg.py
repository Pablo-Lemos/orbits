"""
Data for symbolic regression plot
"""

eq_0 = r"${C_1 m_0 m_1 x / r^3}$"  #r"$(((m0 * m1) * x) * (((1511.1415 / r) / r) / r))$"
eq_1 = r"$x/r$"
eq_2 = r"$x / [r (r + C_1)]$"
eq_3 = r"$(x / ((C_1 + (r / (m0 + m1))) * r))$"
eq_4 = r"${x / (r + C_1)}$"  #r"$(x / (C_1 + r))$"
eq_5 = r"$(((C_1 * m0) * m1) * x)$"
eq_6 = r"$(((m0 * m1) * x) * (C_1 / r))$"
eq_7 = r"${C_1 m_0 m_1 x / r^2}$"  #r"$(((m0 * m1) * x) * ((C_1 / r) / r))$"
eq_8 = r"$(x / (r + (C_1 * m0  * m1)))$"
eq_9 = r"${C_1 x / (C_2 - r)}$"  #r"$(C_1 * (x / (C_2 - r)))$"
eq_10 = r"$((x / ((0.14728448 / (0.3292896 - x)) + r)) / r)$"
eq_11 = r"$(2.1578069 / ((r / x) + (x / (m0 + m1))))$"

comp_0 = 13
comp_1 = 3
comp_2 = 7
comp_3 = 11
comp_4 = 5
comp_5 = 7
comp_6 = 9
comp_7 = 11
comp_8 = 11
comp_9 = 7
comp_10 = 11
comp_11 = 11

eq_list = [eq_1, eq_2, eq_3, eq_4, eq_5, eq_6, eq_7, eq_8, eq_9, eq_10, eq_11, eq_0]
comp_list = [comp_1, comp_2, comp_3, comp_4, comp_5, comp_6, comp_7, comp_8, comp_9, comp_10, comp_11, comp_0]

zipped_lists = zip(comp_list, eq_list)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
comp_list, eq_list = [list(tuple) for tuple in tuples]

eqs_1 = [eq_1, eq_2, eq_3, eq_0]
losses_1 = [4.227e-01, 2.018e-01, 1.260e-01, 7.469e-04]
scores_1 = [4.306e-01, 3.126e-01, 2.322e-01, 2.564e+00]

eqs_2 = [eq_1, eq_4, eq_5, eq_6, eq_7, eq_8]
losses_2 = [6.028e-01, 2.646e-01, 1.137e-01, 3.619e-02, 1.615e-02, 2.202e-05]
scores_2 = [2.531e-01, 4.117e-01, 4.223e-01, 5.725e-01, 4.034e-01, 3.299e+00]

eqs_3 = [eq_1, eq_2, eq_8, eq_0]
losses_3 = [5.842e-01, 4.484e-01, 2.393e-01, 5.695e-04]
scores_3 = [2.688e-01, 1.294e-01, 3.107e-01, 3.020e+00]

eqs_4 = [eq_1, eq_4, eq_9, eq_7, eq_0]
losses_4 = [6.335e-01, 2.634e-01, 3.308e-02, 1.244e-02, 5.972e-04]
scores_4 = [2.283e-01, 4.388e-01, 1.037e+00, 4.311e-01, 1.518e+00]

eqs_5 = [eq_1, eq_4, eq_9, eq_7, eq_0]
losses_5 = [6.624e-01, 1.845e-01, 4.607e-02, 2.860e-02, 7.957e-05]
scores_5 = [2.059e-01, 6.392e-01, 6.937e-01, 2.109e-01, 2.942e+00]

eqs_6 = [eq_1, eq_4, eq_10, eq_0]
losses_6 = [4.904e-01, 3.185e-01, 1.794e-01, 4.679e-04]
scores_6 = [3.563e-01, 2.158e-01, 1.915e-01, 2.975e+00]

eqs_7 = [eq_1, eq_2, eq_11, eq_0]
losses_7 = [4.284e-01, 1.723e-01, 9.337e-02, 1.033e-03]
scores_7 = [4.239e-01, 4.006e-01, 3.061e-01, 2.252e+00]

eqs_8 = [eq_1, eq_2, eq_0]
losses_8 = [4.155e-01, 1.668e-01, 6.404e-04]
scores_8 = [4.391e-01, 3.995e-01, 2.748e+00]

eqs_9 = [eq_1, eq_4, eq_9, eq_6, eq_7, eq_0]
losses_9 = [7.337e-01, 3.297e-01, 8.396e-02, 5.952e-02, 2.750e-02, 5.976e-04]
scores_9 = [1.548e-01, 4.000e-01, 6.839e-01, 1.721e-01, 3.860e-01, 1.915e+00]

eqs_10 = [eq_1, eq_4, eq_2, eq_3, eq_0]
losses_10 = [4.034e-01, 2.778e-01, 1.160e-01, 8.286e-02, 7.693e-04]
scores_10 = [4.539e-01, 1.865e-01, 4.366e-01, 1.559e-01, 2.340e+00]

eqs_all = [eqs_1, eqs_2, eqs_3, eqs_4, eqs_5, eqs_6, eqs_7, eqs_8, eqs_9, eqs_10]
losses_all = [losses_1, losses_2, losses_3, losses_4, losses_5, losses_6, losses_7, losses_8, losses_9, losses_10]
scores_all = [scores_1, scores_2, scores_3, scores_4, scores_5, scores_6, scores_7, scores_8, scores_9, scores_10]

losses_list = []
scores_list = []

for eq in eq_list:
    loss_temp = []
    scores_temp = []
    for i, j in enumerate(eqs_all):
        if eq in j:
            index = j.index(eq)
            loss_temp.append(losses_all[i][index])
            scores_temp.append(scores_all[i][index])

    losses_list.append(loss_temp)
    scores_list.append(scores_temp)