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
def crear_dataset(modelo,estados):
    datos = []
    for i, estado_paso in enumerate(estados):
        for nodo, estado in enumerate(estado_paso):
            datos.append({'Time step': modelo.time[i], 'Node': nodo, 'Device status': estado})
    return pd.DataFrame(datos)
