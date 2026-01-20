#Import dependencies
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from ._helpers import plots_input_validation
from ..metrics.metrics import mean_infected_gillespie , compute_max_statistics

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

def plot_bars(model= None, snapshots = None, time = None , stride=1, state_colors= None,gillespie= True):
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
    stride : int, optional
        Subsampling step applied to snapshots and time, by default 1.
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
        stride=stride,
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

def plot_network_evolution(network,model=None,snapshots=None,time=None,stride =  1, state_colors=None):
    """
    Generate a sequence of interactive Plotly figures showing the evolution 
    of node states in the network across time. Each frame represents a snapshot 
    of the network at one time step.

    Parameters
    ----------
    network : networkx.Graph
        Network whose structure will be visualized.
    model : object, optional
        Model object containing:
        - network_history
        - time
    snapshots : list of lists, optional
        List of node-state snapshots (used if model is None).
    time : list, optional
        Time points corresponding to snapshots.
    stride : int, optional
        Subsampling step applied to snapshots and time, by default 1.
    state_colors : dict
        Mapping from states to colors.
        
    """

    # --- Input validation ---
    snapshots, time, state_colors = plots_input_validation(
        model = model,
        snapshots= snapshots,
        time = time,
        stride= stride,
        state_colors=state_colors
    )
     # Compute fixed layout once for visual consistency
    fixed_layout = nx.spring_layout(network)

    figures = []
    for i in range(len(time)):
        
        

       # Node traces
        node_traces = go.Scatter(
            x=[fixed_layout[node][0] for node in network.nodes()],
            y=[fixed_layout[node][1] for node in network.nodes()],
            mode='markers',
            marker=dict(
                size=10,
                color=[state_colors[state] for state in snapshots[i]],
                line_width=0.5
            ),
            hoverinfo='text',
            text=[f"Node {node}<br>Device status: {snapshots[i][node]}" for node in network.nodes()]
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

def plot_animation(network,model=None,snapshots=None,time=None,state_colors=None,title=None, save_html= False):
    """
    Generate an interactive Plotly animation showing the evolution
    of node states in a network.

    Parameters
    ----------
    network : networkx.Graph
        Network whose structure will be visualized.
    model : object, optional
        Model object containing:
        - network_history
        - time
    snapshots : list of lists, optional
        List of node-state snapshots (used if model is None).
    time : list, optional
        Time points corresponding to snapshots.
    state_colors : dict
        Mapping from states to colors.
    title : str, optional
        Title of the animation.
    """

    # --- Input validation ---
    snapshots, time, state_colors = plots_input_validation(
        model = model,
        snapshots= snapshots,
        time = time,
        stride=1,
        state_colors=state_colors
    )

    fixed_layout = nx.spring_layout(network, seed=42)

    
    frames = []
    for i, snapshot in enumerate(snapshots):
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
        title=title if title is not None else "Stochastic network dynamics",
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
    if save_html:
        fig.write_html("infection_animation.html")
    fig.show()
def plot_hist_I_max(
    model,
    n_sim=100,
    steps=50,
    tmax=100,
    gillespie=True,
    title=None
):
    """
    Plot the histogram of the maximum number of infected nodes (I_max)
    obtained from repeated stochastic simulations.

    Parameters
    ----------
    model : object
        Stochastic model providing:
        - network_history : list of lists with node states
        - time : list of event times or discrete steps.
    n_sim : int, optional
        Number of simulations to run, by default 100.
    steps : int, optional
        Number of Monte Carlo steps (if not using Gillespie), by default 50.
    tmax : float, optional
        Maximum simulation time for Gillespie simulations, by default 100.
    gillespie : bool, optional
        If True, uses Gillespie SSA time; otherwise uses discrete steps.
    title : str, optional
        Custom title for the plot.
    """

    if model is None:
        raise ValueError("A model must be provided.")

    res_dic = compute_max_statistics(
        model=model,
        n_sim=n_sim,
        gillespie=gillespie,
        tmax=tmax,
        steps=steps,
    )

    I_max_list = res_dic["I_max_values"]

    plt.hist(
        I_max_list,
        bins=range(min(I_max_list), max(I_max_list) + 1),
        color='skyblue',
        edgecolor='black'
    )

    plt.xlabel(r"Maximum number of infected nodes ($I_{\max}$)")
    plt.ylabel("Frequency")
    plt.title(title if title is not None else "Distribution of $I_{max}$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_hist_tf(model, n_sim=100, steps=50, tmax=100, gillespie=True):
    """
    Plot histogram of the outbreak duration (time until no infected nodes remain).

    Parameters
    ----------
    model : object
        Model object containing:
        - network_history : list of lists with states ('S', 'I', 'R',...) for each node.
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
    #Input validation
    if model is None:
        raise ValueError("A model must be provided")
    

    res_dic = compute_max_statistics(
        model=model,
        n_sim=n_sim,
        gillespie=gillespie,
        tmax=tmax,
        steps=steps,
    )

    tf_vec = res_dic["extinction_time_values"]
    plt.hist(tf_vec, bins=30, color = 'skyblue', edgecolor='black')
    plt.xlabel('Time until outbreak extinction ($t_{fin}$)')
    plt.ylabel('Frequency')
    plt.title('Distribution of outbreak duration')
    plt.tight_layout()
    plt.show()

def plot_Imax_comparative(
        models,
        tags,
        title,
        n_sim = 100,
        steps = 50,
        tmax= 100,
        gillespie= True,
):
    """
    Plot a comparative histogram of outbreak sizes for multiple models.

    Parameters
    ----------
    models : List of model objects containing:
        - network_history : list of lists with states ('S', 'I', 'R',...) for each node.
        - time : list of event times or discrete steps.
    tags : sequence of str
        Labels associated with each model, used in the plot legend.
    title : str or None
        Title of the figure. If None, a default title is used.
    n_sim : int, optional
        Number of independent simulations run for each model, by default 100.
    steps : int, optional
        Number of Monte Carlo steps for discrete-time simulations, by default 50.
        Ignored if `gillespie` is True.
    tmax : float, optional
        Maximum simulation time for Gillespie simulations, by default 100.
    gillespie : bool, optional
        If True, simulations are run using Gillespie SSA time.
        If False, discrete-time simulations are used.
    """
    plt.figure(figsize=(8,5))
    colors = ["#1f77b4", "#ff7f0e"]
    for model,tag,color in zip(models,tags,colors):
        res_dic = compute_max_statistics(
            model=model,
            n_sim=n_sim,
            tmax=tmax,
            steps= steps,
            gillespie= gillespie
        )
        I_max_vec = res_dic["I_max_values"]
        plt.hist(I_max_vec, bins=range(min(I_max_vec), max(I_max_vec)+1), alpha=0.5, color=color,
                 label=f"{tag}",
                 edgecolor="black")
    plt.xlabel(" ($I_{max}$)")
    plt.title(title if title is not None else "Comparative outbreak size distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()