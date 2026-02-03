import pandas as pd
import numpy as np
import networkx as nx

def network_summary(G:nx.Graph)->dict:
    """
    This function provides a summary of commonly used network-level metrics

    Parameters
    ----------
    G : networkx.Graph
        Undirected network.

    Returns
    -------
    dict
        Dictionary containing network summary statistics:
        
        * 'N' : int  
            Number of nodes in the network.
        * 'avg_k' : float  
            Average node degree.
        * 'C' : float  
            Average clustering coefficient.
        * 'rho_A' : float  
            Spectral radius of the adjacency matrix.

    Notes
    -----
    - The spectral radius is computed as the largest real part of the
      eigenvalues of the adjacency matrix.

    Raises
    ------
    TypeError
        If `G` is not a NetworkX Graph instance.
    """


    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph instance.")
    degs = [d for _, d in G.degree()]
    avg_k = np.mean(degs)
    C = nx.average_clustering(G)
    #L = nx.average_shortest_path_length(max(nx.connected_components(G), key=len), G)
    rho = max(np.linalg.eigvals(nx.to_numpy_array(G))).real
    return {"N": G.number_of_nodes(),
            "avg_k": avg_k,
            "C": C,
            # "L": L,
            "rho(A)": rho}

# Convertir la simulación en un DataFrame
def dataset_creation(model,title,save_csv=False):
    """
    Convert the output of a single simulation into a tidy DataFrame.

    Parameters
    ----------
    model : object
        Model object exposing:
        - network_history : list of node-state snapshots
        - time : list of time points

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns:
        * 'time'
        * 'node'
        * 'state'
    """
    data = []
    for i, snapshot in enumerate(model.network_history):
        for node, state in enumerate(snapshot):
            data.append({'Time step': model.time[i], 'Node': node, 'Device status': state})
    df =  pd.DataFrame(data)
    return df
    