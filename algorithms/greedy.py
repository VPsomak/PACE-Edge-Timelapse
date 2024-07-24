import timeit
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from networkx.algorithms import approximation
from heapq import heapify, heappop
import sys

class Heap():
    """ Representation of a Heap data structure """
    # data format: [node_degree, node_index]
    heap = []
    hash = dict()

    def init(self, initial):
        self.heap = initial
        for value, index in initial:
            self.hash[index] = value
        self.rebuild()

    def rebuild(self):
        heapify(self.heap)

    def pop(self):
        return heappop(self.heap)

    def contains(self, index):
        return index in self.hash

    def update(self, index, value):
        self.hash[index] = value
        for i, e in enumerate(self.heap):
            if e[1] == index:
                self.heap[i] = [value, index]
                break
        self.rebuild()

    def get(self, index):
        return self.hash.get(index)

    def size(self):
        return len(self.heap)

class GreedySolver():
    """ Representation of a greedy solver """
    
    def __init__(self,graph:nx.Graph,scoring:str='degree_full'):
        self.graph = graph
        self.nodes_activated = [node for node,data in self.graph.nodes(data=True) if data['activated']]
        self.nodes_hosting = [node for node,data in self.graph.nodes(data=True) if data['host']]
        self.scoring = scoring
        self.coverset = []
        self.edges = {}
        self.heap:Heap = None
        self.scores = {}

    def get_node_score(self,node):
        """ Gets the score of the node in order to be used in the heap """
        if self.scoring == 'degree_full':
            # Calculate the score based on the node degree using all edges of the node
            return self.graph.degree[node]
        elif self.scoring == 'degree_active':
            # Calculate the score based on the node degree using only edges pointing to activated nodes
            subgraph = self.graph.copy()
            for edge in self.graph.edges():
                if not edge[1] in self.nodes_activated:
                    subgraph.remove_edge(*edge)
            return subgraph.degree[node]

    def build_heap(self,mode):
        """ Creates the heap of nodes sorted by their node score """
        self.heap = Heap()
        self.scores = {}

        data = []  # data format: [node_degree, node_index]
        for node in self.graph.nodes:
            node_index = node
            host_preservation_weight = 1
            # Mode = 2 means that host preservation should be considered
            if node_index in self.nodes_hosting and mode == 2:
               host_preservation_weight = 2
            self.scores[node_index] = self.get_node_score(node_index)*host_preservation_weight
            # multiply to -1 for desc order
            data.append([-1 * self.scores[node_index], node_index])
        self.heap.init(data)

    def place_image(self,node_index):
        """ Places an image to the specified node """
        adj = set(self.graph.edges([node_index]))
        for u, v in adj:
            # remove edge from list
            self.edges.discard((u, v))
            self.edges.discard((v, u))

            # update neighbors
            if self.heap.contains(v):
                new_degree = self.scores[v] - 1
                # update index
                self.scores[v] = new_degree
                # update heap
                self.heap.update(v, -1 * new_degree)
        # add node in mvc
        self.mvc.add(node_index)
        self.coverset.append(self.mvc)

    def solve(self,mode):
        """ Runs the algorithm on the provided graph """
        self.mvc = set()
        subgraph = self.graph.subgraph(self.nodes_activated)
        self.build_heap(mode)
        self.edges = set(subgraph.edges)
        for host in self.nodes_hosting:
            self.place_image(host)

        while len(self.edges) > 0:
            # remove node with max degree
            _, node_index = self.heap.pop()
            self.place_image(node_index)
        return