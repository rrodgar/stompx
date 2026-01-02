import pandas as pd

def resumen_red(G):
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
