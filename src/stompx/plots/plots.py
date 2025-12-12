#Import dependencies
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def plot_infected_curve(simulation:object, gillespie = True):
    """
        Plot the temporal evolution of infected nodes from a simulation.

        Parameters
        ----------
        simulation : object
            A simulation object containing the attributes:

            * network_snapshots : list of lists with the state of each node.
            * time : list of time points at which events occurred.

        gillespie : bool, optional
            If True, the x-axis is labeled as continuous time 't' (Gillespie SSA).
            If False, the x-axis is treated as discrete Monte Carlo steps.
    """
    snapshots = simulation.network_history
    infected_counts_time_step = [snapshot.count('I') for snapshot in snapshots]
    plt.plot(simulation.time, infected_counts_time_step, label='Infected')
    if gillespie:
        plt.xlabel('time')
    else:
        plt.xlabel('MC_step')
    plt.ylabel('I')
    plt.title('Temporal evolution of infected nodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bars(simulation, steps=1, gillespie= True):
    """_summary_

    Parameters
    ----------
    simulation : object
        Simulation object containing:
        - network_history : list of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    steps : int, optional
        Subsampling interval, by default 1
    gillespie : bool, optional
        If True, the x-axis is labeled as continuous time 't' (Gillespie SSA).
        If False, the x-axis is treated as discrete Monte Carlo steps, 
        by default True
    """

    
    state_counts = {'S': [], 'I': [], 'R': []}
    for state_list in simulation.network_history:
        state_counts['S'].append(state_list.count('S'))
        state_counts['I'].append(state_list.count('I'))
        state_counts['R'].append(state_list.count('R'))

    
    idxs = list(range(0, len(simulation.time), steps))
    S_vals = [state_counts['S'][i] for i in idxs]
    I_vals = [state_counts['I'][i] for i in idxs]
    R_vals = [state_counts['R'][i] for i in idxs]

    
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(S_vals)), S_vals, color='blue', label='S')
    plt.bar(range(len(I_vals)), I_vals, color='red', bottom=S_vals, label='I')
    plt.bar(range(len(R_vals)), R_vals, color='green', bottom=np.array(S_vals) + np.array(I_vals), label='R')

    plt.xticks(range(0, len(S_vals), max(1, len(S_vals)//10)), labels=[str(idxs[i]) for i in range(0, len(idxs), max(1, len(idxs)//10))])
    
    if gillespie:
        plt.xlabel('Time')
    else:
        plt.xlabel('MC_Step')
    plt.ylabel('# Nodes')
    plt.title('State Evolution in IoT Network(SIR Model)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_network_evolution(simulation, network, steps):
    """
    Generate a sequence of interactive Plotly figures showing the evolution 
    of node states in the network across time. Each frame represents a snapshot 
    of the network at one time step.

    Parameters
    ----------
    simulation : object
        Simulation object containing:
        - network_history : list of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    network : networkx.Graph
        The underlying network whose structure (nodes and edges) will be plotted.
    steps : int 
        Number of time steps to display. Must not exceed the length of 
        `simulation.network_history`.
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
                color=[state_colors[state] for state in simulation.network_history[i]],
                line_width=0.5
            ),
            hoverinfo='text',
            text=[f"Node {node}<br>Device status: {simulation.network_history[i][node]}" for node in network.nodes()]
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

def plot_animation(simulation:object, network:object)->None:
    """_summary_

    Parameters
    ----------
    simulation : object
        Simulation object containing:
        - network_history : list of lists with states ('S', 'I', 'R') for each node.
        - time : list of event times or discrete steps.
    network : networkx.Graph
        The underlying network whose structure (nodes and edges) will be plotted.
    """

    
    fixed_layout = nx.spring_layout(network, seed=42)

   
    state_colors = {'S': 'blue', 'I': 'red', 'R': 'green'}

    
    frames = []
    for i, snapshot in enumerate(simulation.network_history):
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
        title=f"Stochastic-Network SIR — β = {simulation.beta}, γ = {simulation.gamma}, nI(t0)= {simulation.num_initial_infected}",
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
def plot_hist_I_max(modelo,n_sim=100, steps = 50, tmax = 100 ,gillespie = True):
    suma_tf = 0
    suma_I_max = 0
    I_max_vec = []
    prob_ext = 0
    #iteraciones
    for i in range(n_sim):
        if gillespie:
            modelo.run_simulation(tmax, verbose = False )
        else:
            modelo.run_simulation(steps)
        I = [state.count('I') for state in modelo.network_history]
        I_max = max(I)
        I_max_vec.append(I_max)
        tf = modelo.time[np.where(np.array(I) > 0)[0][-1]] 
        suma_tf += tf
        if I_max < 10:
            prob_ext +=1
        
    I_max_vec = np.array(I_max_vec)
    media_I_max = I_max_vec.mean()
    std_I_max = I_max_vec.std()
    media_tf = suma_tf/n_sim
    # Histograma
    # plt.hist(I_max_vec, bins=30, alpha=0.7, color='red', edgecolor = 'black')
    plt.hist(I_max_vec, bins=range(min(I_max_vec), max(I_max_vec)+1), color = 'skyblue', edgecolor='black')
    plt.xlabel('Número máximo de infectados')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de $I_{{\\max}}$ ({modelo})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f" — I_max medio: {media_I_max:.2f} ± {std_I_max:.2f}", 'tf',tf)
    print('simulationes extinguidas antes del brote',prob_ext)
    return np.unique(I_max_vec)

def plot_hist_tf(modelo, nombre='Modelo', n_sim=100, steps=50, tmax=100, gillespie=True):
    tf_vec = []

    for i in range(n_sim):
        if gillespie:
            modelo.run_simulation(tmax, verbose=False)
        else:
            modelo.run_simulation(steps)

        # Recuento de infectados a lo largo del tiempo
        I = [state.count('I') for state in modelo.network_history]

        # Si hubo algún infectado en la simulación, obtenemos el último tiempo con I > 0
        if any(I):
            tf = modelo.time[np.where(np.array(I) > 0)[0][-1]]
            tf_vec.append(tf)

    tf_vec = np.array(tf_vec)
    media_tf = tf_vec.mean()
    std_tf = tf_vec.std()

    # Histograma
    plt.hist(tf_vec, bins=30, color = 'skyblue', edgecolor='black')
    plt.xlabel('Tiempo hasta la extinción del brote ($t_{\\text{fin}}$)')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de $t_{{\\text{{fin}}}}$ ({nombre})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f" — t_fin medio: {media_tf:.2f} ± {std_tf:.2f}")
    return tf_vec