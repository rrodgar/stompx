#Import dependencies
import matplotlib as plt
import numpy as np

def plot_infected_curve(simulation, gillespie = True):
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

def plot_barras(simulation, steps=1, gillespie= True):
    # Obtener el recuento de cada estado en cada paso de tiempo
    recuento_estados = {'S': [], 'I': [], 'R': []}
    for estado_paso in simulation.network_history:
        recuento_estados['S'].append(estado_paso.count('S'))
        recuento_estados['I'].append(estado_paso.count('I'))
        recuento_estados['R'].append(estado_paso.count('R'))

    # Submuestreo cada 'steps' pasos
    idxs = list(range(0, len(simulation.time), steps))
    S_vals = [recuento_estados['S'][i] for i in idxs]
    I_vals = [recuento_estados['I'][i] for i in idxs]
    R_vals = [recuento_estados['R'][i] for i in idxs]

    # Crear el gráfico de barras con los valores reducidos
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(S_vals)), S_vals, color='blue', label='S')
    plt.bar(range(len(I_vals)), I_vals, color='red', bottom=S_vals, label='I')
    plt.bar(range(len(R_vals)), R_vals, color='green', bottom=np.array(S_vals) + np.array(I_vals), label='R')

    plt.xticks(range(0, len(S_vals), max(1, len(S_vals)//10)), labels=[str(idxs[i]) for i in range(0, len(idxs), max(1, len(idxs)//10))])
    # plt.xlabel('Evento (muestreo cada {} eventos)'.format(steps))
    if gillespie:
        plt.xlabel('Evento')
    else:
        plt.xlabel('pMC')
    plt.ylabel('# de nodos')
    plt.title('Evolución de estados en la red IoT (modelo SIR)')
    plt.legend()
    plt.tight_layout()
    plt.show()