#import dependencies
import networkx as nx
import numpy as np
import random

class Montecarlo_SIR_Network():
    def __init__(self,G,gamma, h,num_initial_infected, initial_inf_method = 'aleatorio'):
        """_summary_

        Parameters
        ----------
        G : _type_
            _description_
        gamma : _type_
            _description_
        h : _type_
            _description_
        num_initial_infected : int
            Initial numbers of infected devices (nodes)
        initial_inf_method : Strategy to select the initial infected nodes:
            * 'random' : random nodes (default)
            * 'degree' : nodes with highest degree
            * 'eigenvector : nodes with highest eigenvector centrality
        """
        #--------------------------------
        # * G: Grafo o red donde propagar la enfermedad. Objeto de NetworkX
        # * beta: Tasa de infección
        # * gamma: Tasa de recuperacion
        # * h : umbral vecinos
        # * alfa: Pendiente funcion logistica
        #--------------------------------
        self.G = G
        self.gamma = gamma
        self.num_initial_infected = num_initial_infected
        self.h = h
        self.initial_inf_method = initial_inf_method
        
    def initialize_network(self):
        # random.seed(5487)
        # Inicializamos toda la red a 'S' y otros a 'I' aleatoriamente
        for n in self.G.nodes:
            self.G.nodes[n]['state'] = 'S'
        #Recibe el numero de infectados iniciales y los elige aleatoriamente
        if self.initial_inf_method == 'eigenvector':
            centralidad = nx.eigenvector_centrality_numpy(self.G)
            node_inicio = max(centralidad, key=centralidad.get)
            # print('el node central',node_max_closeness)
            self.G.nodes[node_inicio]['state'] = 'I'
        elif self.initial_inf_method == 'degree':
            centralidad = nx.degree_centrality(self.G)
            node_inicio = max(centralidad, key = centralidad.get)
            self.G.nodes[node_inicio]['state'] = 'I'
        else:
            for node in random.sample(list(self.G.nodes), k= self.num_initial_infected):
                self.G.nodes[node]['state'] ='I'
    def new_state(self, node, states_old):
        # Recibe un node y calculamos su nuevo state
           
        
        state = states_old[node]
        # Vamos a generar un num aleatorio para las probabilidades
        # Comprobamos el state 
        r = np.random.rand()
        if state =='I':
            return 'R' if (r < self.gamma) else 'I'
        elif state =='S':
            #Calculamos los vecinos infectados
            vecinos_inf = [i for i in self.G.neighbors(node) if states_old[i]=='I']
            kv = len(vecinos_inf)
            # Calculamos la prob de transición
            if kv == 0:
                p = 0
            else:
                p = 1/(1 + np.exp(self.h - self.beta * kv ))            
            return 'I' if (r < p) else 'S'
        
        else:
            return 'R' 
    def run_simulation(self, steps):
        # Ejecutamos la simulación completa
        #Inicializamos la red
        self.initialize_network()

        # Simular un paso monteCarlo pasando por cada node y ver si se actualiza su state.
        self.network_history = []
        self.time = np.arange(steps)
        
        for t in range(steps):

            #Guardamos el state viejo de los nodes
            states_old = {node: self.G.nodes[node]['state'] for node in self.G.nodes() }
            self.network_history.append([self.G.nodes[n]['state'] for n in self.G.nodes() ]) 

            # Calculamos el nuevo state 
            for node in self.G.nodes():
                self.G.nodes[node]['state'] = self.new_state(node,states_old)