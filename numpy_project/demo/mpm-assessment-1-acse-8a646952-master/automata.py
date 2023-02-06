"""Implementations of Lorenz 96 and Conway's
Game of Life on various meshes"""

# from re import X
import numpy as np
# import scipy
import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon


def lorenz96(initial_state, nsteps):
    """
    Perform iterations of the Lorenz 96 update.
    Parameters
    ----------
    initial_state : array_like or list
    Initial state of lattice in an array of floats.
    nsteps : int
    Number of steps of Lorenz 96 to perform.
    Returns
    -------
    numpy.ndarray
    Final state of lattice in array of floats
    >>> x = lorenz96([8.0, 8.0, 8.0], 1)
    >>> print(x)
    array([8.0, 8.0, 8.0])
    >>> lorenz96([False, False, True, False, False], 3)
    array([True, False, True, True, True])
    """
    x = np.array(initial_state*1)
    i = 1
    while i <= nsteps:
        X1 = np.roll(x, 2)
        X2 = np.roll(x, -1)
        X3 = np.roll(x, 1)
        x = (100*x + ((X1 - X2) * X3) + 8) / 101
        i += 1
    return x


def sums_table(initial_state):
    x = np.array(initial_state*1)
    x_down = np.roll(x, 1, axis=0)
    x_down[0:1, :] = 0
    x_up = np.roll(x, -1, axis=0)
    x_up[x_up.shape[0] - 1:x_up.shape[0], :] = 0

    y = x + x_up + x_down

    y_right = np.roll(y, 1, axis=1)
    y_right[:, 0:1] = 0
    y_left = np.roll(y, -1, axis=1)
    y_left[:, y_left.shape[1]-1:y_left.shape[1]] = 0

    sums = y + y_left + y_right - x
    return sums


def life(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
    Initial 2d state of grid in an array of booleans.
    nsteps : int
    Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
    Final state of grid in array of booleans
    """
    i = 1
    x = np.array(initial_state*1)
    while i <= nsteps:
        update_x = np.zeros((x.shape[0], x.shape[1]))
        sums = sums_table(x)
        update_x[(x == 0) & (sums == 3)] = 1
        update_x[(x == 1) & (2 <= sums) & (3 >= sums)] = 1
        x = update_x
        i += 1
    return x.astype(bool)


def sums_table1(initial_state):
    x = np.array(initial_state*1)
    x_down = np.roll(x, 1, axis=0)
    x_up = np.roll(x, -1, axis=0)
    y = x + x_up + x_down
    y_right = np.roll(y, 1, axis=1)
    y_left = np.roll(y, -1, axis=1)
    sums = y + y_left + y_right - x
    return sums


def life_periodic(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life on a doubly periodic mesh.
    Parameters
    ----------
    initial_state : array_like or list of lists
    Initial 2d state of grid in an array of booleans.
    nsteps : int
    Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
    Final state of grid in array of booleans
    """

    i = 1
    x = np.array(initial_state*1)
    while i <= nsteps:
        update_x = np.zeros((x.shape[0], x.shape[1]))
        sums = sums_table1(x)
        update_x[(x == 0) & (sums == 3)] = 1
        update_x[(x == 1) & (2 <= sums) & (3 >= sums)] = 1
        x = update_x
        i += 1

    return x.astype(bool)


def sums_table2(initial_state):

    x = np.array(initial_state)
    x[x == -1] = 1
    x_down = np.roll(x, 1, axis=0)
    x_up = np.roll(x, -1, axis=0)
    y = x + x_up + x_down
    y_right = np.roll(y, 1, axis=1)
    y_left = np.roll(y, -1, axis=1)
    sums = y + y_left + y_right - x
    return sums


def life2colour(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life on a doubly periodic mesh.
    Parameters
    ----------
    initial_state : array_like or list of lists
    Initial 2d state of grid in an array ints with value -1, 0, or 1.
    Values of -1 or 1 represent "on" cells of both colours. Zero
    values are "off".
    nsteps : int
    Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
    Final state of grid in array of ints of value -1, 0, or 1.
    """
    i = 1
    x = np.array(initial_state)
    while i <= nsteps:
        update_x = x.copy()
        sums = sums_table2(x)
        sums1 = sums_table1(x)
        update_x[(x == -1) & (sums != 3) & (sums != 2)] = 0
        update_x[(x == 1) & (sums != 3) & (sums != 2)] = 0
        update_x[(x == 0) & (sums == 3) & (sums1 >= 0)] = 1
        update_x[(x == 0) & (sums == 3) & (sums1 < 0)] = -1
        x = update_x
        i += 1

    return x


def sums_table3(initial_state):

    x = np.array(initial_state*1)
    x_down = np.roll(x, 1, axis=0)
    x_down[0:1, :] = 0
    x_up = np.roll(x, -1, axis=0)
    x_up[x_up.shape[0] - 1:x_up.shape[0], :] = 0
    y = x + x_up + x_down
    y_right = np.roll(y, 1, axis=1)
    y_right[:, 0:1] = 0
    y_left = np.roll(y, -1, axis=1)
    y_left[:, y_left.shape[1] - 1:y_left.shape[1]] = 0
    sums = y + y_left + y_right - x
    sums1 = np.pad(sums, 1)
    x1 = np.pad(x, 1)

    for row, col in np.ndindex(x.shape):

        if (row % 2 == 0) & (col % 2 == 0):
            sums1[row+1, col+1] -= x1[1+row+1, 1+col+1]
        if (row % 2 == 0) & (col % 2 != 0):
            sums1[row+1, col+1] -= x1[1+row-1, 1+1+col]
        if (row % 2 != 0) & (col % 2 == 0):
            sums1[row+1, col+1] -= x1[1+row+1, 1+col-1]
        if (row % 2 != 0) & (col % 2 != 0):
            sums1[row+1, col+1] -= x1[1+row-1, 1+col-1]

    return sums1[1:-1, 1:-1]


def lifepent(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on
    a pentagonal tessellation.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial state of grid of pentagons.
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
         Final state of tessellation.
    """

    # write your code here to replace return this statement
    i = 1
    x = np.array(initial_state*1)
    while i <= nsteps:
        update_x = np.zeros((x.shape[0], x.shape[1]))
        sums = sums_table3(x)
        update_x[(x == 0) & ((sums == 3) | (sums == 4) | (sums == 6))] = 1
        update_x[(x == 1) & (2 <= sums) & (3 >= sums)] = 1
        x = update_x
        i += 1
    return x.astype(bool)


# Remaining routines are for plotting


def plot_lorenz96(data, label=None):
    """
    Plot 1d array on a circle

    Parameters
    ----------
    data: arraylike
        values to be plotted
    label:
        optional label for legend.


    """

    offset = 8

    data = np.asarray(data)
    theta = 2*np.pi*np.arange(len(data))/len(data)

    vector = np.empty((len(data), 2))
    vector[:, 0] = (data+offset)*np.sin(theta)
    vector[:, 1] = (data+offset)*np.cos(theta)

    theta = np.linspace(0, 2*np.pi)

    rings = np.arange(int(np.floor(min(data))-1),
                      int(np.ceil(max(data)))+2)
    for ring in rings:
        plt.plot((ring+offset)*np.cos(theta),
                 (ring+offset)*np.sin(theta), 'k:')

    fig_ax = plt.gca()
    fig_ax.spines['left'].set_position(('data', 0.0))
    fig_ax.spines['bottom'].set_position(('data', 0.0))
    fig_ax.spines['right'].set_color('none')
    fig_ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks(rings+offset, rings)
    plt.fill(vector[:, 0], vector[:, 1],
             label=label, fill=False)
    plt.scatter(vector[:, 0], vector[:, 1], 20)


def plot_array(data, show_axis=False,
               cmap=plt.cm.get_cmap('seismic'), **kwargs):
    """Plot a 1D/2D array in an appropriate format.

    Mostly just a naive wrapper around pcolormesh.

    Parameters
    ----------

    data : array_like
        array to plot
    show_axis: bool, optional
        show axis numbers if true
    cmap : pyplot.colormap or str
        colormap

    Other Parameters
    ----------------

    **kwargs
        Additional arguments passed straight to pyplot.pcolormesh
    """
    plt.pcolormesh(1*data[-1::-1, :], edgecolor='y',
                   vmin=-2, vmax=2, cmap=cmap, **kwargs)

    plt.axis('equal')
    if show_axis:
        plt.axis('on')
    else:
        plt.axis('off')


def plot_pent(x_0, y_0, theta_0, clr=0):
    """
    Plot a pentagram

    Parameters
    ----------
    x_0: float
        x coordinate of centre of the pentegram
    y_0: float
        y coordinate of centre of the pentegram
    theta_0: float
        angle of pentegram (in radians)
    """
    colours = ['w', 'r']
    s_1 = 1/np.sqrt(3)
    s_2 = np.sqrt(1/2)

    theta = np.deg2rad(theta_0)+np.deg2rad([30, 90, 165, 240, 315, 30])
    r_pent = np.array([s_1, s_1, s_2, s_1, s_2, s_1])

    x_pent = x_0+r_pent*np.sin(-theta)
    y_pent = y_0+r_pent*np.cos(-theta)

    plt.fill(x_pent, y_pent, ec='k', fc=colours[clr])


def plot_pents(data):
    """
    Plot pentagrams in Cairo tesselation, coloured by value

    Parameters
    ----------
    data: arraylike
        integer array of values
    """
    plt.axis('off')
    plt.axis('equal')
    data = np.asarray(data).T
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            x_c = (row+1)//2+(row//2)*np.cos(np.pi/6)-(col//2)*np.sin(np.pi/6)
            y_c = (col+1)//2+(col//2)*np.cos(np.pi/6)+(row//2)*np.sin(np.pi/6)
            theta = (90*(row % 2)*((col + 1) % 2)
                     - 90*(row % 2)*(col % 2) - 90*(col % 2))
            clr = data[row, data.shape[1]-1-col]
            plot_pent(x_c, y_c, theta, clr=clr)
