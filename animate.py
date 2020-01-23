import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np

AU = 149.6e6 * 1000     # Astronomical Unit in meters.
RESCALE = 100/AU

save = True #Whether to save the animation
show = True #Whether to show the animation
path = './orbits/'
names = ['Sun', 'Mercury', 'Venus', 'Earth']

# The planet radii. They are all in units of AU. 
# The sizes are scaled by 100 since otherwise they are not visible
# The size of the sun is divided by 4 for aesthetic purposes
sizes = [6.9634e8/4.*RESCALE, 4.879e6*RESCALE, 1.2104e7*RESCALE, 1.2756e7*RESCALE]

colors  = ['yellow','brown','pink','blue']

def read_orbits(path, names):
    """Reads the orbit files

    Parameters: 
    -----------
    path: string
        the path to the orbit files
    names: list
        list of names of planets in orbits path

    Returns: 
    --------
    orbits : list
        a list of trajectories for each body (in AU)
    """
    
    orbits = []
        
    for name in names: 
        orbit = np.loadtxt(path+name+'.dat')
        orbits.append(orbit)

    return orbits

def create_plot(orbits, sizes, colors):
    """Creates the figure instance and the objects to be plotted
    
    Parameters: 
    -----------
    orbits : list
        a list of trajectories to be plotted (in AU)
    sizes : list
        the size of each body (in AU)
    colors : list
        the color of each body

    Returns:
    fig, ax
        The matplotlib figure instances
    lines: list
        The objects to be animated
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create all objects to be drawn
    lines = []
    for i in range(len(orbits)):
        r = sizes[i] # Radius of planet (AU)
        c = colors[i] # Color
        x = orbits[i][0][0] # X coordinate (AU)
        y = orbits[i][0][1] # Y coordinate (AU)
        line = plt.Circle((x,y),r, color = c)
        lines.append(line)

    #Set plot limits
    ax.set_xlim(-1.1, 1.1) #AU
    ax.set_ylim(-1.1, 1.1) #AU

    return fig, ax, lines

def init():
    """initialize animation
    
    Returns
    -------
    Lines: list
        A list of objects to be plotted in their initial positions
    """

    for i in range(nbodies):
        x = orbits[i][0][0] # X coordinate (AU)
        y = orbits[i][0][1] # Y coordinate (AU)
        lines[i].center = (x,y)
        ax.add_artist(lines[i])

    return lines

def animate(i):
    """Move the objects to their position in the next frame

    Parameters
    ----------
    i: int
        the frame number

    Returns
    -------
    lines: list
        a list of objects in each frame
    """

    for j in range(nbodies):
        x = orbits[j][i][0] # X coordinate (AU)
        y = orbits[j][i][1] # Y coordinate (AU)
        lines[j].center = (x,y)

    return lines

orbits = read_orbits(path, names)
nbodies = len(names)
frames = len(orbits[0])

# Start plot
fig, ax, lines = create_plot(orbits, sizes, colors)
# Animate
ani = animation.FuncAnimation(fig, animate, frames=frames,
                              interval=20, blit=False, init_func=init)

# Save animation
if save==True: 

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani.save(path+'orbits.mp4', writer=writer)

# Show the animation
if show==True:
    plt.show()
