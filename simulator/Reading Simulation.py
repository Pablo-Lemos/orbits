import pickle
import numpy as np
import matplotlib.pyplot as plt

GR_file = open('GR_simulation.pickle', 'rb')
GR_system = pickle.load(GR_file)

N_file = open('Newton_simulation.pickle', 'rb')
N_system = pickle.load(N_file)

# Extract the position and velocity
GR_positions = GR_system.get_positions()
GR_v = GR_system.get_velocities()
N_positions = N_system.get_positions()
N_v = N_system.get_velocities()

'''print('GR POSITIONS: ')
print(GR_x)
print('N POSITIONS: ')
print(N_x)
print('Difference: ')
print(GR_x - N_x)'''

#time = GR_x[:, 0]
GR_positions_mercury = GR_positions[:, 1]
N_positions_mercury = N_positions[:, 1]
GR_positions_mercury_x = GR_positions_mercury[:, 0]
N_positions_mercury_x = N_positions_mercury[:, 0]
GR_positions_mercury_y = GR_positions_mercury[:, 1]
N_positions_mercury_y = N_positions_mercury[:, 1]
delta_y = N_positions_mercury_y - GR_positions_mercury_y
delta_x = N_positions_mercury_x - GR_positions_mercury_x
#GR_positions_x = GR_positions[:, 0]
#N_positions_x = N_positions[:, 0]

'''plt.plot(GR_positions_mercury_x, GR_positions_mercury_y, label='GR Mercury')
plt.plot(N_positions_mercury_x, N_positions_mercury_y, label='Newtonian Mercury')
plt.ylabel('Y position')
plt.xlabel('X position')
plt.title('Newtonian Positions vs GR Positions')
plt.legend()
plt.show()'''

plt.plot(delta_x, delta_y)
plt.ylabel('Y position')
plt.xlabel('X position')
plt.title('Newtonian and GR difference')
plt.show()


