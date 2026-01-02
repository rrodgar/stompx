#import dependencies
import networkx as nx
import numpy as np
import random

class Montecarlo_SIR_Network():
    def __init__(self,G,gamma, h,num_initial_infected, initial_inf_method = 'aleatorio'):
        """
        Stochastic SIR model on a static network using a discrete Montecarlo scheme.

        Parameters
        ----------
        G : networkx.Graph
            Network where the epidemic spreads (undirected graph)
        gamma : float
            Recovery rate, probability that a node becomes recovered from infected state, values range between [0,1]
        h : int
            Infection threshold parameter. If infected neighbors are higher probability of infection increases.
        num_initial_infected : int
            Initial numbers of infected devices (nodes)
        initial_inf_method : Strategy to select the initial infected nodes:
            * 'random' : random nodes (default)
            * 'degree' : nodes with highest degree
            * 'eigenvector : nodes with highest eigenvector centrality
        """
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
            self.G.nodes[node_inicio]['state'] = 'I'
        elif self.initial_inf_method == 'degree':
            centralidad = nx.degree_centrality(self.G)
            node_inicio = max(centralidad, key = centralidad.get)
            self.G.nodes[node_inicio]['state'] = 'I'
        else:
            for node in random.sample(list(self.G.nodes), k= self.num_initial_infected):
                self.G.nodes[node]['state'] ='I'
    def new_state(self, node, states_old):
    
        state = states_old[node]
        r = np.random.rand()
        if state =='I':
            return 'R' if (r < self.gamma) else 'I'
        elif state =='S':
            
            inf_neighbors = [i for i in self.G.neighbors(node) if states_old[i]=='I']
            kv = len(inf_neighbors)
            if kv == 0:
                p = 0
            else:
                p = 1/(1 + np.exp(self.h  - kv ))            
            return 'I' if (r < p) else 'S'
        
        else:
            return 'R' 
    def run_simulation(self, steps):
       
        self.initialize_network()
        self.network_history = []
        self.time = np.arange(steps)
        
        for t in range(steps):

            
            states_old = {node: self.G.nodes[node]['state'] for node in self.G.nodes() }
            self.network_history.append([self.G.nodes[n]['state'] for n in self.G.nodes() ]) 
            for node in self.G.nodes():
                self.G.nodes[node]['state'] = self.new_state(node,states_old)

class Montecarlo_SIR_Network_TOM(Montecarlo_SIR_Network):
    def TOM(self,node_i:int,node_j:int)->float:
        """ Compute the topological overlap measure for two nodes

        Parameters
        ----------
        node_i : int
            Node number 'i' in the networkx graph G
        node_j : int
            Node number 'j' in the networkx graph G

        Returns
        -------
        float
            TOM value for nodes i and j
        """
        
        neighbors_i = set(self.G.neighbors(node_i))
        neighbors_j = set(self.G.neighbors(node_j))
        l_ij = len(neighbors_i & neighbors_j)
        k_i = self.G.degree(node_i)
        k_j = self.G.degree(node_j)
        tom = (l_ij + 1) / (min(k_i, k_j))
        return tom
    def new_state(self, node, states_old):
        state = states_old[node]
        r = np.random.rand()
        if state =='I':
            return 'R' if (r < self.gamma) else 'I'
        elif state =='S':
            inf_neighbors = [i for i in self.G.neighbors(node) if states_old[i]=='I']
            kv = len(inf_neighbors)
            if kv == 0:
                p = 0
            else:
                for n in inf_neighbors:
                    kv += self.TOM(node,n)
                p = 1/(1 + np.exp(self.h -  kv ))            
            return 'I' if (r < p) else 'S'
        
        else:
            return 'R' 