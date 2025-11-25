def plot_curva_infectados(simulacion, gillespie = True):
    """
        Plot the temporal evolution of infected nodes from a simulation.
    """
    hist = simulacion.historial_red
    infectados_por_tiempo = [snapshot.count('I') for snapshot in hist]
    plt.plot(simulacion.time,infectados_por_tiempo, label='Infectados')
    if gillespie:
        plt.xlabel('t')
    else:
        plt.xlabel('pMC')
    plt.ylabel('I')
    plt.title('Evolución de infectados')
    plt.legend()
    plt.grid(True)
    plt.show()