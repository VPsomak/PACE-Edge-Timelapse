#	This file is part of Distributed Image Placer.
#
#    Distributed Image Placer is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Distributed Image Placer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Distributed Image Placer.  If not, see https://www.gnu.org/licenses/.

import random, sys
from algorithms import ilp, approx, greedy, genetic
import maxmin
import networkx as nx
import numpy as np
import numpy.random
import time
from math import ceil
from utils import Visualizer
from pathlib import Path
# import satispy

class PACE():
    """ Representation of the PACE-Edge main class """
    def __init__(self,*args,**kwargs):
        random.seed(kwargs.get('seed',10))
        numpy.random.seed(kwargs.get('seed',10))
        sys.setrecursionlimit(kwargs.get('recursionlimit',100000))
        self.imageSize = kwargs.get('imageSize',3*1024*1024*1024)
        # 10737418240
        self.bandwidthEthernet = kwargs.get('bandwidthEthernet',10*1024*1024*1024)
        # 26214400
        self.bandwidthWifi = kwargs.get('bandwidthWifi',25*1024*1024)
        # 524288
        self.bandwidthlocalfile = kwargs.get('bandwidthlocalfile',0.5*1024*1024)
        self.hw_fault_probability = kwargs.get('hw_fault_probability',0.0)
        self.placementCost = self.imageSize / self.bandwidthWifi
        self.activated_ratio = kwargs.get('activated_ratio',1.0)
        self.name = kwargs.get('name','grapher')
        self.graph = kwargs.get('graph',None)
        self.pos = kwargs.get('pos',None) # Used for stabilizing the graph visualization between runs
        self.previous_hosts = [
            node for node,data in self.graph.nodes(data=True) if data['host']
        ] if self.graph else []
        self.shortest_paths = nx.shortest_path(self.graph) if self.graph else None
        self.solution_text = None # The solution text after solving
        try:
            self.model = kwargs['model']
        except:
            raise Exception('Available models [ilp, approximation, greedy, genetic]')
        try:
            self.graph_type = kwargs['graph_type']
        except:
            raise Exception('Available graphs [binomial_tree, balanced_tree, star, barabasi_albert, erdos_renyi, newman_watts_strogatz]')
        
    def get_cost(self):
        """ Returns the cost value by applying the cost function """
        preserved_hosts = [
            node for node,data in self.graph.nodes(data=True) if data['host'] and (node in self.previous_hosts)
        ]
        new_hosts = [
            node for node,data in self.graph.nodes(data=True) if data['host'] and (node not in self.previous_hosts)
        ]
        cost = (len(new_hosts) * self.placementCost) + (len(preserved_hosts) * (self.placementCost/4)) + \
            sum([
                self.graph.edges[edge[0],edge[1]]['usage'] / \
                self.graph.edges[edge[0],edge[1]]['capacity'] \
                for edge in self.graph.edges
            ])
        return cost

    def create_continuum(self,size=64, degree=3, branching_factor_of_tree=4, height_of_tree=4, knearest=7, probability=0.7):
        """ Creates a semi-randomized graph based on the provided parameters """

        if self.graph_type =="binomial_tree":
            self.graph = nx.generators.classic.binomial_tree(size)
        elif self.graph_type =="balanced_tree":
            # balanced_tree(r, h, create_using=None),
            # r - Branching factor of the tree; each node will have r children.
            # h - Height of the tree.
            self.graph = nx.generators.classic.balanced_tree(branching_factor_of_tree, height_of_tree)
        elif self.graph_type =="star":
            self.graph = nx.star_graph(size)
        elif self.graph_type =="barabasi_albert":
            # barabasi_albert_graph(n, m, seed=None)
            # n: Number of nodes
            # m: Number of edges to attach from a new node to existing nodes
            self.graph = nx.barabasi_albert_graph(size, degree)
        elif self.graph_type =="erdos_renyi":
            # erdos_renyi_graph(n, p, seed=None, directed=False)
            # n: Number of nodes
            # p: Probability of edge creation
            self.graph = nx.erdos_renyi_graph(size, probability, seed=None, directed=False)
        elif self.graph_type =="newman_watts_strogatz":
            # n: The number of nodes.
            # k: Each node is joined with its k nearest neighbors in a ring topology.
            # p: The probability of adding a new edge for each edge.
            self.graph = nx.newman_watts_strogatz_graph(size, knearest, probability, seed=None)
        else:
            raise Exception("Available graphs [binomial_tree, balanced_tree, star, barabasi_albert, erdos_renyi, newman_watts_strogatz]")

        print("--Vertices:", len(self.graph.nodes), "--Edges:", len(self.graph.edges), "\n")

        NODES = self.graph.number_of_nodes()
        nodes_activated = np.random.choice(NODES, ceil(NODES*self.activated_ratio), replace=False)
        nx.set_node_attributes(self.graph, values=False, name='activated')
        nx.set_node_attributes(self.graph, values={node:True for node in nodes_activated}, name='activated')
        nx.set_node_attributes(self.graph, values=False, name='host')
        nx.set_node_attributes(self.graph, values=False, name='offline')

        # Attributes to the graph
        edgeCapacities = {}
        for edge in self.graph.edges:
            if edge[0] == edge[1]:
                edgeCapacities[edge] = self.bandwidthlocalfile
            elif random.random() < 0.7:
                edgeCapacities[edge] = self.bandwidthWifi
            else:
                edgeCapacities[edge] = self.bandwidthEthernet

        # max-min fairness
        output = maxmin.max_min_fairness(demands=list(edgeCapacities.values()), capacity=20000000000)
        counter = 0
        for key, value in edgeCapacities.items():
            edgeCapacities[key] = output[counter]
            counter = counter + 1
        # End of max-min fairness

        nx.set_edge_attributes(self.graph, values=edgeCapacities, name='capacity')
        nx.set_edge_attributes(self.graph, values=0, name='usage')
        nx.set_edge_attributes(self.graph, values=0, name='time')
        nx.set_edge_attributes(self.graph, values=0, name='numImages')
        self.shortest_paths = nx.shortest_path(self.graph)

    def add_hw_faults(self):
        """ Method that emulates hardware faults by cutting of the communication (edges) of random nodes """
        online_nodes = [node for node,data in self.graph.nodes(data=True) if not data['offline']]
        for node in online_nodes:
            if random.random() < self.hw_fault_probability:
                edges = list(self.graph.edges(node))
                nx.set_node_attributes(self.graph, values={node:True}, name='offline')
                nx.set_node_attributes(self.graph, values={node:edges}, name='offline_edges')
                nx.set_node_attributes(self.graph, values={node:False}, name='activated')
                nx.set_node_attributes(self.graph, values={node:False}, name='host')
                self.graph.remove_edges_from(edges)

    def restore_hw_faults(self):
        """ Method that emulates hardware restoration by restoring the communication (edges) of random offline nodes """
        node_data_dict = dict(self.graph.nodes(data=True))
        offline_nodes = [node for node in node_data_dict if node_data_dict[node]['offline']]
        for node in offline_nodes:
            if random.random() < 0.5:
                edges = self.graph.edges(node)
                nx.set_node_attributes(self.graph, values={node:False}, name='offline')
                self.graph.add_edges_from(node_data_dict[node]['offline_edges'])
                nx.set_node_attributes(self.graph, values={node:[]}, name='offline_edges')

    def update_continuum(self):
        """ Updates the graph for the current timestep if a historical graph is provided """
        
        
        self.restore_hw_faults()
        
        online_nodes = [node for node,data in self.graph.nodes(data=True) if not data['offline']]
        NODES = len(online_nodes)
        nodes_activated = np.random.choice(online_nodes, ceil(NODES*self.activated_ratio), replace=False)
        nx.set_node_attributes(self.graph, values=False, name='activated')
        nx.set_node_attributes(self.graph, values={node:True for node in nodes_activated}, name='activated')
        self.add_hw_faults()

        # Attributes to the graph
        edgeCapacities = {}
        for edge in self.graph.edges:
            if edge[0] == edge[1]:
                edgeCapacities[edge] = self.bandwidthlocalfile
            elif random.random() < 0.7:
                edgeCapacities[edge] = self.bandwidthWifi
            else:
                edgeCapacities[edge] = self.bandwidthEthernet

        # max-min fairness
        output = maxmin.max_min_fairness(demands=list(edgeCapacities.values()), capacity=20000000000)
        counter = 0
        for key, value in edgeCapacities.items():
            edgeCapacities[key] = output[counter]
            counter = counter + 1
        # End of max-min fairness

        nx.set_edge_attributes(self.graph, values=edgeCapacities, name='capacity')
        nx.set_edge_attributes(self.graph, values=0, name='usage')
        nx.set_edge_attributes(self.graph, values=0, name='time')
        nx.set_edge_attributes(self.graph, values=0, name='numImages')
        self.shortest_paths = nx.shortest_path(self.graph)

    def solve(self):
        """ 
        Solves the problem using the provided graph and algorithm. 
        The solution is stored in the self.graph variable using the node data 'host' value 
        and the edge data 'time' and 'usage' values. 
        The total cost can be accessed using the self.get_cost() method.
        """
        if self.graph is None:
            self.create_continuum()
        else:
            self.update_continuum()

        nodes_activated = [node for node,data in self.graph.nodes(data=True) if data['activated']]

        start_time = time.time()

        if self.model == "ilp":
            ilpModel = ilp.ilp_model(self.graph,self.imageSize)
            res = ilpModel.solve()
            if res['statusCode'] == 1:
                nodes_with_image = []
                for var in res['variables']:
                    if 'activation' in var.name:
                        if var.value() > 0:
                            nodes_with_image.append(int(var.name.split('_')[1]))
                    else:
                        nodes = var.name.split('_')[1]+var.name.split('_')[2]
                        n = int(nodes.split(',')[0].replace('(',''))
                        d = int(nodes.split(',')[1].replace(')',''))
                        self.graph[n][d]['usage'] = var.value()
                        self.graph[n][d]['numImages'] = round(self.graph[n][d]['usage'] / self.imageSize,4)
                        self.graph[n][d]['time'] = round(self.graph[n][d]['usage'] / self.graph[n][d]['capacity'],6) * 100
            else:
                print(res['status'])
                return
        elif self.model == "approximation":
            approx_solver = approx.ApproxSolver(self.graph)
            approx_solver.solve()
            res = approx_solver.coverset
            nodes_with_image = []
            if len(res) > 0:
                nodes_with_image = res[0]
            nearest_image = []
            for active_node in nodes_activated:
                paths = []
                for host in nodes_with_image:
                    if host in self.shortest_paths[active_node]:
                        paths.append(self.shortest_paths[active_node][host])
                if len(paths) > 0:
                    nearest_image.append(min(paths, key=lambda x: len(x)))
                else:
                    nearest_image.append(None)
            for i in range(len(nodes_activated)):
                if nearest_image[i] is not None:
                    sp = (nearest_image[i])
                    for j in range(len(sp) - 1):
                        self.graph[sp[j]][sp[j + 1]]['usage'] += self.imageSize
                        self.graph[sp[j]][sp[j + 1]]['numImages'] = round(self.graph[sp[j]][sp[j + 1]]['usage'] / self.imageSize, 4)
                        self.graph[sp[j]][sp[j + 1]]['time'] = self.graph[sp[j]][sp[j + 1]]['usage'] / self.graph[sp[j]][sp[j + 1]]['capacity']

        elif self.model == "greedy":
            res = []
            # greedy.minimum_vertex_cover_hybrid_greedy(self.graph, res)
            greedy_solver = greedy.GreedySolver(self.graph)
            greedy_solver.solve()
            res = greedy_solver.coverset
            nodes_with_image = []
            if len(res) > 0:
                nodes_with_image = res[0]
            nearest_image = []
            for active_node in nodes_activated:
                paths = []
                for host in nodes_with_image:
                    if host in self.shortest_paths[active_node]:
                        paths.append(self.shortest_paths[active_node][host])
                if len(paths) > 0:
                    nearest_image.append(min(paths, key=lambda x: len(x)))
                else:
                    nearest_image.append(None)
            for i in range(len(nodes_activated)):
                if nearest_image[i] is not None:
                    sp = (nearest_image[i])
                    for j in range(len(sp) - 1):
                        self.graph[sp[j]][sp[j + 1]]['usage'] += self.imageSize
                        self.graph[sp[j]][sp[j + 1]]['numImages'] = round(self.graph[sp[j]][sp[j + 1]]['usage'] / self.imageSize, 4)
                        self.graph[sp[j]][sp[j + 1]]['time'] = self.graph[sp[j]][sp[j + 1]]['usage'] / self.graph[sp[j]][sp[j + 1]]['capacity']
        elif self.model == "genetic":
            genetic_solver = genetic.GeneticSolver(self.graph,self.imageSize)
            genetic_solver.solve()
            res = genetic_solver.coverset
            nodes_with_image = []
            if len(res) > 0:
                nodes_with_image = res[0]
            transfered = res[2]
            for edge in transfered:
                self.graph[edge[0]][edge[1]]['usage'] = transfered[edge]
                self.graph[edge[0]][edge[1]]['numImages'] = round(self.graph[edge[0]][edge[1]]['usage'] / self.imageSize,4)
                self.graph[edge[0]][edge[1]]['time'] = round(self.graph[edge[0]][edge[1]]['usage'] / self.graph[edge[0]][edge[1]]['capacity'],6) * 100
        else:
            print("Give a correct model as indicated from the list\n")
            print("Available models [ilp, approximation, genetic, greedy] \n")
            sys.exit()

        nx.set_node_attributes(self.graph, values=False, name='host')
        nx.set_node_attributes(self.graph, values={node:True for node in nodes_with_image}, name='host')
        online_nodes = [node for node,data in self.graph.nodes(data=True) if not data['offline']]
        offline_nodes = [node for node,data in self.graph.nodes(data=True) if data['offline']]

        self.solution_text = f"Execution Time: {round(time.time() - start_time,4)} seconds"
        self.solution_text += f"\nModel: {len(self.model)}"
        self.solution_text += f"\nGraph: {len(self.graph_type)}"
        self.solution_text += f"\nTotal nodes: {len(self.graph.nodes)}"
        self.solution_text += f"\nOnline nodes: {len(online_nodes)}"
        self.solution_text += f"\nOffline nodes: {len(offline_nodes)}"
        self.solution_text += f"\nNodes with image: {len(nodes_with_image)}"
        self.solution_text += f"\nNodes activated: {len(nodes_activated)}"
        self.solution_text += f"\nCost: {round(self.get_cost(),4)}"

        vis = Visualizer(graph=self.graph,title=f"Placement with {self.model} algorithm",legend=self.solution_text)
        Path(f"graphs/{self.model}/reducted").mkdir(parents=True, exist_ok=True)
        self.pos = vis.visualize_full(filename=f"graphs/{self.model}/{self.name}_{self.model}_{self.graph_type}_full.jpg",pos=self.pos)

        subgraph = self.graph.copy()
        edges_to_remove = [(u, v) for u, v, d in self.graph.edges(data=True) if d['time'] == 0]
        subgraph.remove_edges_from(edges_to_remove)
        vis = Visualizer(graph=subgraph,title=f"Placement with {self.model} algorithm",legend=self.solution_text)
        vis.visualize_full(filename=f"graphs/{self.model}/reducted/{self.name}_{self.model}_{self.graph_type}_reduced.jpg")


        return {
            'time':round(time.time() - start_time,4),
            'nodes':len(self.graph.nodes),
            'online_nodes':len(online_nodes),
            'offline_nodes':len(offline_nodes),
            'hosts':len(nodes_with_image),
            'activated':len(nodes_activated),
            'cost':round(self.get_cost(),4)
        }