
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

    def can_be_infection(self, edge:tuple[int,int]):
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
            node_inicio = max(centralidad, key=centralidad.get)
            self.G.nodes[node_inicio]['state'] = 'I'

        elif self.initial_inf_method == 'degree':
            centralidad = nx.degree_centrality(self.G)
            node_inicio = max(centralidad, key = centralidad.get)
            self.G.nodes[node_inicio]['state'] = 'I'

        else:
            for node in random.sample(list(self.G.nodes), k= self.num_initial_infected):
                self.G.nodes[node]['state'] ='I'

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


    def remove_edges_SI(self,node):
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
        for neighbor in self.G.neighbors(node):
            par = tuple(sorted((node, neighbor)))
            if par in self.E_SI:
                to_remove.add(par)
        self.E_SI.difference_update(to_remove)
    def add_edges_SI(self,node):
        """
        Add all new S-I edges created after a node becomes infected.

        Parameters
        ----------
        node : int
            Newly infected node
        """
        for neighbor in self.G.neighbors(node):
            if self.G.nodes[neighbor]['state'] == 'S':
                par = tuple(sorted((node, neighbor)))
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
        """
        Run an exact stochastic simulation of the Gillespie algorithm

        Parameters
        ----------
        tmax : float
            Maximum simulation time.
        verbose : bool, optional
            If True, print intermediate messages about the simulation status.
        """
        self.initialize_network()
        t = 0
        events = 0
        self.time = [t]
        self.network_history = [] 
        self.network_history.append([self.G.nodes[node]['state'] for node in self.G.nodes()]) 
        while (t < tmax):
            self.Prob_rates()
            if self.w_i[-1] == 0:
                if verbose:
                    print(f"No susceptible or infected nodes remain. "f"Simulation ended at t = {self.time[-1]}")
                break

            r = np.random.rand()
            dt = -1/self.w_i[-1] * np.log(r)
            t += dt

            self.choose_reaction()
            events +=1

            self.time.append(t)
            self.network_history.append([self.G.nodes[node]['state'] for node in self.G.nodes()])
        
        if verbose:
            print('Number of events:', events)