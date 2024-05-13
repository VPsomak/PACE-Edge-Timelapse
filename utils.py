import networkx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class TextLegendHandler(Line2D):
    def __init__(self, text, *args, **kwargs):
        super().__init__([], [], *args, **kwargs)
        self.text = text

    def get_label(self):
        return self.text

class Visualizer():
    """ Class that handles graph visualization """

    def __init__(self,graph,active_nodes=[],hosts=[],layout=networkx.spring_layout,title:str = None, legend:str = None):
        self.graph = graph
        self.hosts = hosts
        #self.active_nodes = {node for node in active_nodes if node not in hosts}
        self.active_nodes = active_nodes
        self.layout = layout
        self.title = title
        self.legend = legend
        self.figsize = (19.2*2, 10.8*2)
        self.node_color = 'skyblue'
        self.edge_color = 'black'
        self.node_size = 1000
        self.linewidths = 1
        self.font_size = 15
    
    def visualize_subset(self,nodes:list,filename:str = None):
        """ Visualize a subset of the graph """
        plt.figure(figsize=self.figsize)
        subgraph = self.graph.subgraph(nodes)
        pos = self.layout(subgraph,weight=lambda edge, data: (1.0 / data['time']) if data['time'] else 1.0)
        networkx.draw(
            subgraph, 
            pos, 
            with_labels=True, 
            node_color=self.node_color, 
            node_size=self.node_size, 
            edge_color=self.edge_color, 
            linewidths=self.linewidths, 
            font_size=self.font_size
        )
        intersecting = [value for value in nodes if value in self.hosts]
        if len(intersecting) > 0 and len(intersecting) < len(nodes):
            networkx.draw_networkx_nodes(subgraph, pos, nodelist=intersecting, node_color='red', node_size=self.node_size)
        intersecting = [value for value in nodes if value in self.active_nodes]
        if len(intersecting) > 0:
            networkx.draw_networkx_nodes(subgraph, pos, nodelist=intersecting, node_color='green', node_size=self.node_size)
        if self.legend:
            legend_elements = [TextLegendHandler(self.legend)]
            plt.legend(handles=legend_elements,loc='lower right')
        if self.title:
            plt.title(self.title)
        if filename:
            plt.savefig(filename)
        #plt.show()
        plt.close()

    def visualize_full(self,filename:str = None,pos = None):
        """ Visualize the complete graph """
        plt.figure(figsize=self.figsize)
        if pos is None:
            pos = self.layout(self.graph,seed=10)
        networkx.draw(
            self.graph, 
            pos, 
            with_labels=True, 
            node_color=self.node_color, 
            node_size=self.node_size, 
            edge_color=self.edge_color, 
            linewidths=self.linewidths, 
            font_size=self.font_size
        )
        only_host_nodes = {node for node in self.hosts if node not in self.active_nodes}
        if len(only_host_nodes) > 0 and len(only_host_nodes) < len(list(self.graph)):
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=only_host_nodes, node_color='red', node_size=self.node_size)
        only_active_nodes = {node for node in self.active_nodes if node not in self.hosts}
        if len(only_active_nodes) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=only_active_nodes, node_color='green', node_size=self.node_size)
        combination_nodes = {node for node in self.active_nodes if node in self.hosts}
        if len(combination_nodes) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=combination_nodes, node_color='orange', node_size=self.node_size)
        edge_labels = {(u, v): f"{round(d['time'],2)}" for u, v, d in self.graph.edges(data=True) if d['time'] > 0}
        networkx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        if self.legend:
            legend_elements = [TextLegendHandler(self.legend)]
            plt.legend(handles=legend_elements,loc='lower right')
        if self.title:
            plt.title(self.title)
        if filename:
            plt.savefig(filename)
        #plt.show()
        plt.close()
        return pos

    def visualize_hosts(self,filename:str = None):
        """ Visualize the hosting nodes only """
        return self.visualize_subset(self.hosts)
    
    def visualize_hosts(self,filename:str = None):
        """ Visualize the active nodes only """
        return self.visualize_subset(self.hosts+self.active_nodes)