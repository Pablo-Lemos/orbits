import pickle
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


GR_file = open('GR_simulation.pickle', 'rb')
GR_system = pickle.load(GR_file)

N_file = open('newton_simulation.pickle', 'rb')
N_system = pickle.load(N_file)

# Extract the position, velocity, and names
x_GR = GR_system.get_positions()
x_N = N_system.get_positions()
names = N_system.get_names()

fig = plt.figure(figsize=(8, 8))

delta_x = x_GR[:, 1, 0] - x_N[:, 1, 0]
delta_y = x_GR[:, 1, 1] - x_N[:, 1, 1]
#time = np.arange(len(delta_x))

GR_sun = x_GR[:, 0, :] - x_GR[:, 0, :]
GR_mercury = x_GR[:, 1, :] - x_GR[:, 0, :]
x_GR_rel = np.stack([GR_sun, GR_mercury], axis=1)
N_sun = x_N[:, 0, :] - x_N[:, 0, :]
N_mercury = x_N[:, 1, :] - x_N[:, 0, :]
x_N_rel = np.stack([N_sun, N_mercury], axis=1)

'''
ax = plt.axes(projection='3d')
ax.plot3D(GR_sun[:, 0], GR_sun[:, 1], GR_sun[:, 2], 'yo', label='Sun')
#ax.plot3D(GR_mercury[:90, 0], GR_mercury[:90, 1], GR_mercury[:90, 2], label='GR Mercury 1st period')
#ax.plot3D(N_mercury[:90, 0], N_mercury[:90, 1], N_mercury[:90, 2], label='N Mercury 1st period')
ax.plot3D(GR_mercury[:, 0], GR_mercury[:, 1], GR_mercury[:, 2], label='GR Mercury')
ax.plot3D(N_mercury[:, 0], N_mercury[:, 1], N_mercury[:, 2], label='N Mercury')

#ax.title('Mercury\'s Orbit using newtoninan and GR (Beta equation)')
ax.set_zlabel('Z [AU]')
'''

for i in range(2):
    colors_GR = ['yellow', 'red']
    colors_N = ['yellow', 'grey']
    #plt.plot(x_N_rel[:, i, 0], x_N_rel[:, i, 1], color=colors_N[i], label=f'{names[i]} Newtonian')
    plt.plot(x_GR_rel[:, i, 0], x_GR_rel[:, i, 1], color=colors_N[i], label=f'{names[i]} Post Newtonian Augmented')
    #plt.plot(x_N[23000:23046, i, 0], x_N[23000:23046, i, 1], label=f'{names[i]} N middle period')
    #plt.plot(x_GR[-368:, i, 0], x_GR[-368:, i, 1], 'o', color=colors_GR[i], label=f'{names[i]} GR simulation')
    #plt.plot(x_N[-368:, i, 0], x_N[-368:, i, 1], 'o', color=colors_N[i], label=f'{names[i]} Newtonian simulation')

#plt.plot(time, delta_y, label= 'Time vs Delta Y (GR-N)')

"""
plt.plot(GR_sun[:, 0], GR_sun[:, 1], 'yo', label=f'SUN')
plt.plot(GR_mercury[:, 0], GR_mercury[:, 1], color='grey',  label=f'Mercury Post Newtonian Aug')
#plt.plot(N_mercury[:, 0], N_mercury[:, 1], color='grey',  label=f'Mercury Newtonian')
#plt.plot(GR_mercury[-90:, 0], GR_mercury[-90:, 1], label=f'Mercury GR 1000th period')
#plt.plot(N_mercury[-89:, 0], N_mercury[-89:, 1], label=f'Mercury N 1000th period')
"""


plt.legend(loc=4)
plt.xlabel('X [AU]')
#plt.xlabel('Time [arbitrary units]')
plt.ylabel('Y [AU]')

#plt.title('Mercury\'s orbit for 20 earth years using N')
plt.show()



''''# Extract the position and velocity
GR_positions = GR_system.get_positions()
GR_v = GR_system.get_velocities()
N_positions = N_system.get_positions()
N_v = N_system.get_velocities()'''


"""
GR_positions_mercury = GR_positions[:, 1]
N_positions_mercury = N_positions[:, 1]
GR_positions_mercury_x = GR_positions_mercury[:, 0]
N_positions_mercury_x = N_positions_mercury[:, 0]
GR_positions_mercury_y = GR_positions_mercury[:, 1]
N_positions_mercury_y = N_positions_mercury[:, 1]
delta_y = N_positions_mercury_y - GR_positions_mercury_y
delta_x = N_positions_mercury_x - GR_positions_mercury_x
#GR_positions_x = GR_positions[:, 0]
#N_positions_x = N_positions[:, 0]"""

'''plt.plot(GR_positions_mercury_x, GR_positions_mercury_y, label='GR Mercury')
plt.plot(N_positions_mercury_x, N_positions_mercury_y, label='Newtonian Mercury')
plt.ylabel('Y position')
plt.xlabel('X position')
plt.title('Newtonian Positions vs GR Positions')
plt.legend()
plt.show()'''

"""
plt.plot(delta_x, delta_y)
plt.ylabel('Y position')
plt.xlabel('X position')
plt.title('Newtonian and GR difference')
plt.show()"""


