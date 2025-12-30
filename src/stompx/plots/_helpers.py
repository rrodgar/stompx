def plots_input_validation(model = None, snapshots = None, time = None, stride = 1, state_colors = None):
    """
    Validate inputs for plotting functions.

    Parameters
    ----------
    model : object, by default None
        model object containing:
        - network_history : list   of lists with states ('S', 'I', 'R',...) for each node.
        - time : list of event times or discrete steps.
    snapshots : list of lists, optional
        List of node-state snapshots (used if model is None).
    time : list, optional
        Time points corresponding to snapshots.
    stride : int, optional
        Subsampling interval, by default 1
    state_colors : dict
        Mapping from states to colors.

    Returns
    -------
    snapshots : list of lists
    time : list
    state_colors : dict

    Raises
    ------
    ValueError
        If neither a model nor both snapshots and time are provided
    """

    # --- Input validation ---
    if model is not None:
        snapshots = model.network_history[::stride]
        time = model.time[::stride]
    else:
        if snapshots is None or time is None:
            raise ValueError(
                "Either a model must be provided, or both snapshots and time must be specified."
            )
        snapshots = snapshots[::stride]
        time = time[::stride]
    if state_colors is None:
        state_colors = {'S': 'blue', 'I': 'red', 'R': 'green'}
    
    return snapshots, time, state_colors