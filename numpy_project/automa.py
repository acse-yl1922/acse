import numpy as np

def Lorenz96(initial_state, nsteps):
    """
        Perform iterations of the Lorenz 96 update.
        Xi_new = 1/101(100Xi + (Xi-2 + Xi+1)Xi-1 + 8)
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
    initial_state = np.array(initial_state).astype('float')
    n = len(initial_state)-1
    for i in range (nsteps):
        for j in range(n):
            initial_state[j] = (1/101)*(
                                        100*initial_state[j%n] 
                                        + (initial_state[(j-2)%n]-initial_state[(j+1)%n])*initial_state[(j-1)%n]
                                        + 8)
    return initial_state