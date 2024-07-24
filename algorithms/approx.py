import networkx as nx

class ApproxSolver():
    """ Representation of an approximation solver """
    def __init__(self,graph:nx.Graph):
        self.graph = graph
        self.size = []
        self.coverset = []
        self.nodes_activated = [
            node for node,data in self.graph.nodes(data=True) if data['activated']
        ]
        self.nodes_hosting = [
            node for node,data in self.graph.nodes(data=True) if data['host']
        ]

    def solve(self,mode):
        """ Run the solver on the provided graph """
        subgraph = self.graph.copy()
        for edge in self.graph.edges():
            if not (edge[1] in self.nodes_activated or edge[0] in self.nodes_activated):
                subgraph.remove_edge(*edge)
        edges = subgraph.edges
        self.size = []
        self.coverset = []
        s = 0
        cover_ = []
        for edge in edges:
            if edge[0] not in cover_ and edge[1] not in cover_:
                cover_.append(edge[0])
                cover_.append(edge[1])
                s += 2
        self.size.append(s)
        self.coverset.append(cover_)
