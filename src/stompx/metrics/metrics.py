#Funcion que me da la media del numero maximo de infectados tras realizar nsim
def I_max_mean(modelo, n_sim = 100, maxt = 30, pasos = 30, gillespie = True ):
    maxI_suma = 0
    maxI_suma2 = 0
    for i in range(n_sim):
        if gillespie:
            modelo.run_simulation(maxt, verbose = False )
        else:
            modelo.run_simulation(pasos)
        I = [estado.count('I') for estado in modelo.historial_red]
        maxi = max(I)
        maxI_suma += maxi
        maxI_suma2 += maxi * maxi 
    
    I_max_prom = maxI_suma/n_sim
    I_max_var = ((maxI_suma2 - n_sim * (I_max_prom ** 2)) / (n_sim - 1))**(0.5)
    return I_max_prom, I_max_var