import numpy as np
import networkx as nx
from networkx import cycles


def get_rand_graph(n, k=0.5, l=10):
    '''
        Compute a random adjacency matrix and graph with n nodes
        k: probability each edge exists (probability of being an InSAR pair)
        l: temporal baselines to omit


        returns graph and adjacency matrix
    '''
    G = nx.Graph()
    G.add_nodes_from(range(0, n))
    A = np.ones((n, n))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                A[i, j] = 0
            if j > i:
                if np.random.rand(1) > k and np.abs(i - j) > 2:
                    A[i, j] = 0
                    A[j, i] = 0
                elif np.abs(i - j) > n-l:
                    A[i, j] = 0
                    A[j, i] = 0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                G.add_edge(i, j, length=np.abs(i - j))

    A = A + np.eye(n)

    return G, A


def cycle_rank(G):
    '''
        Given a graph G, compute its cycle rank 
        returns cycle rank
    '''
    return np.array(cycles.cycle_basis(G)).shape[0]
