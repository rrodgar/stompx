#Import dependencies
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from ._helpers import plots_input_validation

def plot_infected_curve(model:object, gillespie = True):
    """
        Plot the temporal evolution of infected nodes from a model.

        Parameters
        ----------
        model : object
            A model object containing the attributes:

            * network_snapshots : list of lists with the state of each node.
            * time : list of time points at which events occurred.

        gillespie : bool, optional
            If True, the x-axis is labeled as continuous time 't' (Gillespie SSA).
            If False, the x-axis is treated as discrete Monte Carlo steps.
    """
    snapshots = model.network_history
    infected_counts_time_step = [snapshot.count('I') for snapshot in snapshots]
    plt.plot(model.time, infected_counts_time_step, label='Infected')
    if gillespie:
        plt.xlabel('time')
    else:
        plt.xlabel('MC_step')
    plt.ylabel('I')
    plt.title('Temporal evolution of infected nodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bars(model= None, snapshots = None, time = None , steps=1, state_colors= None,gillespie= True):
    """
    Plot a stacked bar chart showing the temporal evolution of node states.

    Parameters
    ----------
    model : object, optional
        Model object containing:
        - network_history : list of lists with node states
        - time : list of event times or discrete steps.
        If provided, snapshots and time are extracted from the model.
    snapshots : list of lists, optional
        List of node-state snapshots (used if model is None).
    time : list, optional
        Time points corresponding to snapshots (used if model is None).
    steps : int, optional
        Subsampling interval applied to snapshots and time, by default 1.
    state_colors : dict, optional
        Mapping from states to colors. Keys define which states are plotted.
        If None, a default SIR color mapping is used.
    gillespie : bool, optional
        If True, the x-axis is labeled as continuous time.
        If False, the x-axis is labeled as discrete Monte Carlo steps.
    """
    # --- Input validation ---
    snapshots, time, state_colors = plots_input_validation(
        model = model,
        snapshots= snapshots,
        time = time,
        steps=steps,
        state_colors=state_colors
    )
        
    state_counts={state:[] for state in state_colors.keys()}
    #Count different states in each valid snapshot
    for snapshot in snapshots:
        for state in state_colors.keys():
            state_counts[state].append(snapshot.count(state))
     #Figure creation       
    plt.figure(figsize=(15, 6))
    x = range(len(time))
    bottom = np.zeros(len(time))
    for state, color in state_colors.items():
        plt.bar(x, state_counts[state], color=color, bottom=bottom, label=state)
        bottom += np.array(state_counts[state])
    
    
    plt.xlabel('Time' if gillespie else 'MC_Step')
    plt.ylabel('# Nodes')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_network_evolution(model, network, steps):
    """
    Generate a sequence of interactive Plotly figures showing the evolution 
    of node states in the network across time. Each frame represents a snapshot 
    of the network at one time step.

    Parameters
    ----------
    model : object
        model object containing:
        - network_history : list   of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    network : networkx.Graph
        The underlying network whose structure (nodes and edges) will be plotted.
    steps : int 
        Number of time steps to display. Must not exceed the length of 
        `model.network_history`.
    """
     # Compute fixed layout once for visual consistency
    fixed_layout = nx.spring_layout(network)

    figures = []
    for i in range(steps):
        
        state_colors = {'S': 'blue', 'I': 'red', 'R': 'green'}

       # Node traces
        node_traces = go.Scatter(
            x=[fixed_layout[node][0] for node in network.nodes()],
            y=[fixed_layout[node][1] for node in network.nodes()],
            mode='markers',
            marker=dict(
                size=10,
                color=[state_colors[state] for state in model.network_history[i]],
                line_width=0.5
            ),
            hoverinfo='text',
            text=[f"Node {node}<br>Device status: {model.network_history[i][node]}" for node in network.nodes()]
        )

        # Edge traces
        edge_traces = []
        for edge in network.edges():
            x0, y0 = fixed_layout[edge[0]]
            x1, y1 = fixed_layout[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='black'),
                hoverinfo='none'
            ))

        # Build figure
        fig = go.Figure(data=[*edge_traces, node_traces])

        
        fig.update_layout(
            # title=f"Time step {i}",
            showlegend=False,
            hovermode='closest',
            width=600,  
            height=600,  
            font=dict(size=10)  
        )

        
        figures.append(fig)

    # # Display all frames
    for fig in figures:
        fig.show()

def plot_animation(model:object, network:object)->None:
    """Generate an interactive Plotly animation showing the evolution 
    of node states in the network across time. Each frame represents a snapshot 
    of the network at one time step.

    Parameters
    ----------
    model : object
        model object containing:
        - network_history : list of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    network : networkx.Graph
        The underlying network whose structure (nodes and edges) will be plotted.
    """

    
    fixed_layout = nx.spring_layout(network, seed=42)

   
    state_colors = {'S': 'blue', 'I': 'red', 'R': 'green'}

    
    frames = []
    for i, snapshot in enumerate(model.network_history):
        node_traces = go.Scatter(
            x=[fixed_layout[n][0] for n in network.nodes()],
            y=[fixed_layout[n][1] for n in network.nodes()],
            mode='markers',
            marker=dict(
                color=[state_colors[state] for state in snapshot],
                size=10,
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            text=[f'Node {n}<br>Status: {snapshot[j]}' for j, n in enumerate(network.nodes())]
        )

        edge_traces = [go.Scatter(
            x=[fixed_layout[u][0], fixed_layout[v][0], None],
            y=[fixed_layout[u][1], fixed_layout[v][1], None],
            mode='lines',
            line=dict(color='black', width=1),
            hoverinfo='none'
        ) for u, v in network.edges()]

        frames.append(go.Frame(data=edge_traces + [node_traces], name=str(i)))

    
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
        title=f"Stochastic-Network SIR — β = {model.beta}, γ = {model.gamma}, nI(t0)= {model.num_initial_infected}",
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(label='Play', method='animate', args=[None])]
            )],
            sliders=[{
                'steps': [{
                    'args': [[str(i)], {'frame': {'duration': 100, 'redraw': False},
                                        'mode': 'immediate'}],
                    'label': f'{i}',
                    'method': 'animate'
                } for i in range(len(frames))],
                'transition': {'duration': 0},
                'x': 0, 'y': -0.1,
                'len': 1.0
            }]
        ),
        frames=frames
    )

    fig.update_layout(
        showlegend=False,
        width=650,
        height=650,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    #fig.write_html("animacion_infeccion.html")
    fig.show()
def plot_hist_I_max(model,n_sim=100, steps = 50, tmax = 100 ,gillespie = True):
    """_summary_

    Parameters
    ----------
     model : object
        model object containing:
        - network_history : list of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    n_sim : int, optional
        Number of simulations to run, by default 100.
    steps : int, optional
        Number of discrete Monte Carlo steps (if not using Gillespie), by default 50.
    tmax : int, optional
        Maximum time for Gillespie simulation, by default 100.
    gillespie : bool, optional
        If True, uses Gillespie SSA time; otherwise uses discrete steps, by default True.
    """
    sum_tf = 0
    I_max_list = []
    extinction_count = 0
    #iteraciones
    for i in range(n_sim):
        if gillespie:
            model.run_simulation(tmax, verbose = False )
        else:
            model.run_simulation(steps)
        I = [state.count('I') for state in model.network_history]
        I_max = max(I)
        I_max_list.append(I_max)
        tf = model.time[np.where(np.array(I) > 0)[0][-1]] 
        sum_tf += tf
        if I_max < 10:
            extinction_count +=1
        
    I_max_list = np.array(I_max_list)
    mean_I_max = I_max_list.mean()
    std_I_max = I_max_list.std()
    mean_duration = sum_tf/n_sim
    # Histograma
    # plt.hist(I_max_list, bins=30, alpha=0.7, color='red', edgecolor = 'black')
    plt.hist(I_max_list, bins=range(min(I_max_list), max(I_max_list)+1), color = 'skyblue', edgecolor='black')
    plt.xlabel('Maximum number of infected nodes (I_max)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of I_max for {model}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Mean I_max: {mean_I_max:.2f} ± {std_I_max:.2f}")
    print(f"Mean duration until last infection: {mean_duration:.2f}")
    print(f"Number of simulations that went extinct early: {extinction_count}")

def plot_hist_tf(model, n_sim=100, steps=50, tmax=100, gillespie=True):
    """
    Plot histogram of the outbreak duration (time until no infected nodes remain).

    Parameters
    ----------
    model : object
        Model object containing:
        - network_history : list of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    n_sim : int, optional
        Number of simulations to run, by default 100.
    steps : int, optional
        Number of discrete Monte Carlo steps (if not using Gillespie), by default 50.
    tmax : int, optional
        Maximum time for Gillespie simulation, by default 100.
    gillespie : bool, optional
        If True, uses Gillespie SSA time; otherwise uses discrete steps, by default True.
    """
    tf_vec = []

    for i in range(n_sim):
        if gillespie:
            model.run_simulation(tmax, verbose=False)
        else:
            model.run_simulation(steps)

        
        I = [state.count('I') for state in model.network_history]

        # Only consider simulations where infection occurred.
        if any(I):
            tf = model.time[np.where(np.array(I) > 0)[0][-1]]
            tf_vec.append(tf)

    tf_vec = np.array(tf_vec)
    mean_duration = tf_vec.mean()
    std_tf = tf_vec.std()

    
    plt.hist(tf_vec, bins=30, color = 'skyblue', edgecolor='black')
    plt.xlabel('Time until outbreak extinction ($t_{fin}$)')
    plt.ylabel('Frequency')
    plt.title('Distribution of outbreak duration')
    plt.tight_layout()
    plt.show()

    print(f"Mean outbreak duration: {mean_duration:.2f} ± {std_tf:.2f}")

    #return tf_vec