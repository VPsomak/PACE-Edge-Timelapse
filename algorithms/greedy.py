import timeit
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from networkx.algorithms import approximation
from heapq import heapify, heappop
import sys

class Heap():
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

def get_node_score(graph,node,mode='degree_full'):
    """ Gets the score of the node in order to be used in the heap """
    if mode == 'degree_full':
        return graph.degree[node]
    elif mode == 'degree_active':
        activated_dict = dict(graph.nodes(data="activated"))
        nodes_activated = [node for node in activated_dict if activated_dict[node]]
        subgraph = graph.copy()
        for edge in graph.edges():
            if not edge[1] in nodes_activated:
                subgraph.remove_edge(*edge)
        return subgraph.degree[node]

def build_heap(graph):
    heap = Heap()
    score_index = {}

    data = []  # data format: [node_degree, node_index]
    for node in graph.nodes:
        node_index = node
        score_index[node_index] = get_node_score(graph,node_index,mode='degree_active')
        # multiply to -1 for desc order
        data.append([-1 * score_index[node_index], node_index])
    heap.init(data)

    return heap, score_index

def place_image(graph,node_index,heap,mvc,coverset,edges,scores):
    adj = set(graph.edges([node_index]))
    for u, v in adj:
        # remove edge from list
        edges.discard((u, v))
        edges.discard((v, u))

        # update neighbors
        if heap.contains(v):
            new_degree = scores[v] - 1
            # update index
            scores[v] = new_degree
            # update heap
            heap.update(v, -1 * new_degree)
    # add node in mvc
    mvc.add(node_index)
    coverset.append(mvc)

def minimum_vertex_cover_greedy(graph, coverset):
    mvc = set()
    nodes_activated = [node for node,data in graph.nodes(data=True) if data['activated']]
    nodes_hosting = [node for node,data in graph.nodes(data=True) if data['host']]
    subgraph = graph.subgraph(nodes_activated)
    heap, scores = build_heap(graph)
    edges = set(subgraph.edges)
    for host in nodes_hosting:
        place_image(graph,host,heap,mvc,coverset,edges,scores)

    while len(edges) > 0:
        # remove node with max degree
        _, node_index = heap.pop()
        place_image(graph,node_index,heap,mvc,coverset,edges,scores)
    return

###########################################################
###########################################################
def get_degrees(graph):
    degrees = {}

    for node in graph.nodes:
        node_index = node
        degree = graph.degree[node_index]
        degrees[node_index] = degree

    return degrees


def get_heap(nodes, degrees, visited):
    heap = Heap()
    heap_data = []  # data format: [node_degree, node_index]
    for node in nodes:
        if not visited[node]:
            degree = degrees[node]
            # multiply to -1 for desc order
            heap_data.append([-1 * degree, node])
    heap.init(heap_data)

    return heap



def remove_edges_and_update_degrees(edges_to_remove, edges, degrees, visited):
    for u, v in edges_to_remove:
        # remove edge from list
        edges.discard((u, v))
        edges.discard((v, u))
        # update degree
        degrees[v] -= 1
        if degrees[v] == 0:
            visited[v] = True

def minimum_vertex_cover_hybrid_greedy(graph, coverset):
    mvc = set()
    visited = {}

    degrees = get_degrees(graph)
    edges = set(graph.edges)
    nodes = set(graph.nodes)

    # mark node with degree 1 as visited, otherwise not visited
    for node in nodes:
        # init status
        visited[node] = False
        if degrees[node] == 1:
            # mark node as visited
            visited[node] = True
            # remove edges and update node degrees
            for u, v in graph.edges([node]):
                if degrees[v] > 1:
                    # remove edge from list
                    edges.discard((u, v))
                    edges.discard((v, u))
                    # update degree
                    degrees[v] -= 1
                    if degrees[v] == 0:
                        visited[v] = True

    # build heap with nodes not visited
    heap = get_heap(nodes, degrees, visited)

    # heap update factor
    heap_update_factor = sys.maxsize
    total_nodes = heap.size()
    ratio = total_nodes / len(edges)
    if len(nodes) > 100:
        heap_update_factor = int(total_nodes * ratio)

    # greedy
    count = 0
    while(len(edges) > 0):
        count += 1
        # verify if must update heap
        if count > heap_update_factor:
            count = 0
            # build heap with nodes not visited
            heap = get_heap(nodes, degrees, visited)

        try:
            _, node_index = heap.pop()
            if not visited[node_index]:
                visited[node_index] = True
                mvc.add(node_index)
                # remove edges
                remove_edges_and_update_degrees(graph.edges([node_index]), edges, degrees, visited)
        except IndexError:
            # no more nodes
            break
    coverset.append(mvc)
    return mvc