
#Import dependencies
import networkx as nx
import numpy as np
import random

class Gillespie_SIR_Network():
    def __init__(self, G, beta, gamma, num_initial_infected, initial_inf_method = 'aleatorio'):
        """
        Stochastic SIR model on a static network using the Gillespie algorithm.

        Parameters
        ----------
        G : networkx.Graph
            Network where the epidemic spreads (undirected graph)
        beta : float
            Infection rate
        gamma : float
            Recovery rate
        num_initial_infected : int
            Initial numbers of infected devices (nodes)
        initial_inf_method : Strategy to select the initial infected nodes:
            * 'random' : random nodes (default)
            * 'degree' : nodes with highest degree
            * 'eigenvector : nodes with highest eigenvector centrality

        """
        self.G = G
        self.beta = beta
        self.gamma = gamma
        self.num_initial_infected = num_initial_infected
        self.initial_inf_method = initial_inf_method

    def can_be_infection(self, edge):
        """
        Check whether an edge is a susceptible-infected (S-I) contact.

        Parameters
        ----------
        edge : tuple(int, int)
            Edge connecting two adjacent nodes.

        Returns
        -------
        bool
            True if one node is S and the other is I 
        """
        risk_contact = False
        state_src = self.G.nodes[edge[0]]['state']
        state_dst = self.G.nodes[edge[1]]['state']
        
        if (state_src == 'S' and state_dst =='I') or (state_src == 'I' and state_dst =='S'):
            risk_contact = True
        return risk_contact
    def initialize_network(self):
        """
        Set all nodes to Susceptible and infect some initial nodes according
        to the chosen initialization strategy 
        """

        for n in self.G.nodes:
            self.G.nodes[n]['state'] = 'S'
        
        if self.initial_inf_method == 'eigenvector':
            centralidad = nx.eigenvector_centrality_numpy(self.G)
            nodo_inicio = max(centralidad, key=centralidad.get)
            self.G.nodes[nodo_inicio]['state'] = 'I'

        elif self.initial_inf_method == 'degree':
            centralidad = nx.degree_centrality(self.G)
            nodo_inicio = max(centralidad, key = centralidad.get)
            self.G.nodes[nodo_inicio]['state'] = 'I'

        else:
            for nodo in random.sample(list(self.G.nodes), k= self.num_initial_infected):
                self.G.nodes[nodo]['state'] ='I'

        self.initialize_SI_edges()

    def count_infected(self):
        """Update the list of infected nodes and compute its length"""

        self.Infected = [n for n in self.G.nodes if self.G.nodes[n]['state']=='I']
        self.nI = len(self.Infected)

    def initialize_SI_edges(self):
        """ 
        Compute all susceptible-infected (S-I) edges at initialization

        Notes
        -----
        This method is called only once, before the simulation loop.
        Afterward, the S–I edge set is updated dynamically through
        `remove_edges_SI` and `add_edges_SI`.
        """

        self.count_infected()
        self.E_SI = set(e for e in self.G.edges if self.can_be_infection(e))
        
    def find_Susceptible(self,edge):
        """
        Find the suceptible node on a S-I edge

        Parameters
        ----------
        edge: tuple(int,int)
            Edge connecting two nodes

        Returns
        -------
        int
            The node index corresponding to the susceptible node
        
        Raises
        ------
        ValueError
            If the provided edge is not an S-I edge


        """
        state_src = self.G.nodes[edge[0]]['state']
        state_dst = self.G.nodes[edge[1]]['state']
        if (state_src == 'S') and (state_dst == 'I'):
            return edge[0]
        elif (state_src == 'I') and (state_dst == 'S'):
            return edge[1]
        else:
            raise ValueError(f"Edge {edge} does not connect a susceptible and an infected node.")


    def remove_edges_SI(self,nodo):
        """
        Remove all S–I edges incident to a given node.

        Parameters
        ----------
        node : int
            Node whose incident S–I edges will be removed.

        Notes
        -----
        This is used when a node changes state (e.g., S→I or I→R).
        """
        to_remove = set()
        for neighbor in self.G.neighbors(nodo):
            par = tuple(sorted((nodo, neighbor)))
            if par in self.E_SI:
                to_remove.add(par)
        self.E_SI.difference_update(to_remove)
    def add_edges_SI(self,nodo):
        """
        Add all new S-I edges created after a node becomes infected.

        Parameters
        ----------
        node : int
            Newly infected node
        """
        for neighbor in self.G.neighbors(nodo):
            if self.G.nodes[neighbor]['state'] == 'S':
                par = tuple(sorted((nodo, neighbor)))
                self.E_SI.add(par)
        
    def Prob_rates(self):
        """
        Compute the infection and recovery propensities for the Gillespie algorithm
        """
        self.recovery_rate = self.gamma * self.nI
        self.infection_rate = self.beta * len(self.E_SI)
        self.a_i = np.array([self.recovery_rate, self.infection_rate])
        self.w_i = np.cumsum(self.a_i) 
    def choose_reaction(self):
        """
        Choose the following event
        """
        r = self.w_i[-1]*np.random.rand()
        
        if (r < self.w_i[0]): 
            #Recovery event
            self.G.nodes[self.Infected[0]]['state'] = 'R'
            self.remove_edges_SI(self.Infected[0])
            del self.Infected[0]
            self.nI -=1
            
            
        else: 
            # Infection event
            # A susceptible node becomes infected. We must remove S-I edges and recompute new ones.
            
            random_edge = random.choice(list(self.E_SI))
            S_node = self.find_Susceptible(random_edge)
            self.remove_edges_SI(S_node)
            self.G.nodes[S_node]['state']='I'
            self.add_edges_SI(S_node)
            self.Infected.append(S_node)
            self.nI +=1

            

    def run_simulation(self, tmax, verbose = False):
        # Hacemos la simulacion del algortimo de GIllespie para el grafo
        self.initialize_network()
        t = 0
        eventos = 0
        self.time = [t]
        self.historial_red = [] # Lista vacia para almacenar state de cada nodo en un paso de tiempo
        self.historial_red.append([self.G.nodes[nodo]['state'] for nodo in self.G.nodes()]) # Se guarda el primer state
        while (t < tmax):
       # Se calculan los diferentes ratios de prob llamando al método
            self.Prob_rates()
            if self.w_i[-1] == 0:
                if verbose:
                    print(f"El número de susceptibles e infectados es nulo. Simulación finalizada en t = {self.time[-1]}")
                break

            # Generamos numero aleatorio y calculamos incremento temporal mediante dist. Exp:
            r = np.random.rand()
            dt = -1/self.w_i[-1] * np.log(r)
            t += dt

            #Elegimos que suceso ocurre:
            self.choose_reaction()
            eventos +=1

            # Guardamos tiempo y los valores de los nodos
            self.time.append(t)
            self.historial_red.append([self.G.nodes[nodo]['state'] for nodo in self.G.nodes()])
        
        if verbose:
            print('Han ocurrido ', eventos ,'eventos')