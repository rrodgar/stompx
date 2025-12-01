import pandas as pd
# Convertir la simulación en un DataFrame
def crear_dataset(modelo,estados):
    datos = []
    for i, estado_paso in enumerate(estados):
        for nodo, estado in enumerate(estado_paso):
            datos.append({'Time step': modelo.time[i], 'Node': nodo, 'Device status': estado})
    return pd.DataFrame(datos)
