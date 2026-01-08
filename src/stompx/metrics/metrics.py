#Import dependencies
import networkx as nx
from scipy.interpolate import interp1d
import numpy as np
def I_max_mean(model, n_sim = 100, maxt = 30, steps = 30, gillespie = True ):
    """
    Compute the mean and standard deviation of the maximum number of infected
    nodes across multiple stochastic simulations.

    The function runs the same model multiple times and records the peak number
    of infected nodes (state 'I') reached in each simulation

    Parameters
    ----------
    model : object
        Model object containing:
        - network_history : list of lists with states ('S', 'I', 'R',...) for each node.
        - time : list of event times or discrete steps.
    n_sim : int, optional
        Number of independent simulations, by default 100.
    maxt : float, optional
        Maximum simulation time for continuous-time (Gillespie) simulations,
        by default 30.
    steps : int, optional
        Number of Monte Carlo steps for discrete-time simulations, by default 30.
    gillespie : bool, optional
        If True, runs a continuous-time Gillespie simulation.
        If False, runs a discrete-time Monte Carlo simulation.

    Returns
    -------
    I_max_mean : float
        Mean value of the maximum number of infected nodes.
    I_max_std : float
        Standard deviation of the maximum number of infected nodes.
    """

    maxI_sum = 0
    maxI_sum2 = 0
    for i in range(n_sim):
        if gillespie:
            model.run_simulation(maxt, verbose = False )
        else:
            model.run_simulation(steps)
        I = [estado.count('I') for estado in model.network_history]
        maxi = max(I)
        maxI_sum += maxi
        maxI_sum2 += maxi * maxi 
    
    I_max_prom = maxI_sum/n_sim
    I_max_std = ((maxI_sum2 - n_sim * (I_max_prom ** 2)) / (n_sim - 1))**(0.5)
    return I_max_prom, I_max_std
def interpolate_infected(original_times, original_infected, t_max, n_points=100):
    """
    Interpolate the number of infected nodes onto a uniform time grid.

    This function is intended for continuous-time stochastic simulations
    (e.g., Gillespie algorithm), where event times are irregular. It maps
    the original time series of infected counts onto a fixed time grid,
    enabling averaging and comparison across multiple realizations.

    Parameters
    ----------
    original_times : array-like
        Original (irregular) time points at which infection counts were recorded.
    original_infected : array-like
        Number of infected nodes corresponding to each original time point.
    t_max : float
        Maximum simulation time.
    n_points : int, optional
        Number of points in the uniform time grid, by default 100.

    Returns
    -------
    uniform_times : numpy.ndarray
        Uniformly spaced time points between 0 and t_max.
    interpolated_infected : numpy.ndarray
        Interpolated number of infected nodes at the uniform time points.
    """

    uniform_times = np.linspace(0, t_max, n_points)

    interpolator = interp1d(
        original_times,
        original_infected,
        kind='previous',
        fill_value="extrapolate"
    )

    interpolated_infected = interpolator(uniform_times)

    return uniform_times, interpolated_infected
def mean_infected_gillespie(model, n_sim=100, t_max=50, n_points=100):
    """
    Compute the mean and standard deviation of the number of infected nodes
    over time for Gillespie simulations.

    The function runs multiple independent Gillespie simulations, interpolates
    the number of infected nodes onto a uniform time grid, and computes the
    mean and standard deviation across realizations.

    Parameters
    ----------
    model : object
        Gillespie model object containing:
        - network_history : list of lists with states ('S', 'I', 'R',...) for each node.
        - time : list of event times or discrete steps.
    n_sim : int, optional
        Number of independent simulations, by default 100.
    t_max : float, optional
        Maximum simulation time, by default 50.
    n_points : int, optional
        Number of points in the uniform time grid, by default 100.

    Returns
    -------
    uniform_times : numpy.ndarray
        Uniform time grid.
    I_mean : numpy.ndarray
        Mean number of infected nodes at each time point.
    I_std : numpy.ndarray
        Standard deviation of the number of infected nodes at each time point.
    """

    infected_sum = 0.0
    infected_sum2 = 0.0

    for _ in range(n_sim):
        model.run_simulation(tmax=t_max, verbose=False)

        infected = [state.count('I') for state in model.network_history]

        uniform_times, infected_interp = interpolate_infected(
            model.time,
            infected,
            t_max=t_max,
            n_points=n_points
        )

        infected_sum += infected_interp
        infected_sum2 += infected_interp**2

    I_mean = infected_sum / n_sim
    I_std = np.sqrt(infected_sum2 / n_sim - I_mean**2)

    return uniform_times, I_mean, I_std

def compute_max_statistics(
    model,
    n_sim=100,
    tmax=100,
    steps=50,
    target_state='I',
    extinction_threshold=1,
    gillespie=True
):
    """
    Compute outbreak size statistics from repeated stochastic simulations.

    Parameters
    ----------
    model : object, optional
        Model object containing:
        - network_history
        - time
    n_sim : int, optional
        Number of independent simulations.
    tmax : float, optional
        Maximum simulation time (Gillespie).
    steps : int, optional
        Number of Monte Carlo steps (discrete-time).
    target_state : str, optional
        State considered to compute metrics (by default 'I' as infected).
    extinction_threshold : int, optional
        Threshold on I_max to classify early extinction.
    gillespie : bool, optional
        If True, uses Gillespie SSA time; otherwise uses discrete steps, by default True.

    Returns
    -------
    dict
        Dictionary containing outbreak statistics.
            * I_max_values: numpy.ndarray
                Maximum number of infected nodes observed in each simulation.
            * I_max_mean : float
                Mean value of the maximum number of infected nodes.
            * I_max_std : float
                Standard deviation of the maximum number of infected nodes.
            * extinction_prob : float
                Fraction of simulations classified as early extinctions.
            * mean_extinction_time : float
                Mean time until extinction or end of the outbreak.
    """

    I_max_values = []
    tf_values = []
    extinction_count = 0
    # tf_sum = 0.0

    for _ in range(n_sim):

        if gillespie:
            model.run_simulation(tmax, verbose=False)
        else:
            model.run_simulation(steps)

        I = [snap.count(target_state) for snap in model.network_history]
        I_max = max(I)
        I_max_values.append(I_max)

        if I_max <= extinction_threshold:
            extinction_count += 1

        nonzero = np.where(np.array(I) > 0)[0]
        if len(nonzero) > 0:
            tf_values.append(model.time[nonzero[-1]])
            # tf_sum += model.time[nonzero[-1]]
            # tf_sum2 += (model.time[nonzero[-1]])**2

    I_max_values = np.array(I_max_values)
    tf_values = np.array(tf_values)
    # tf_mean = tf_sum / n_sim
    # tf_std =  np.sqrt(tf_sum2 / n_sim - tf_mean**2)

    return {
        "I_max_values": I_max_values,
        "I_max_mean": I_max_values.mean(),
        "I_max_std": I_max_values.std(),
        "extinction_prob": extinction_count / n_sim,
        "extinction_time_values": tf_values,
        "mean_extinction_time": tf_values.mean(),
        "std_extincion_time": tf_values.std()
    }