import networkx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

class TextLegendHandler(HandlerBase):
    def __init__(self, text, fontsize, *args, **kwargs):
        self.text = text
        self.fontsize = fontsize
        super().__init__(*args, **kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        txt = plt.Text(xdescent + width / 2, ydescent + height / 2, self.text,
                       fontsize=self.fontsize, ha="left", va="bottom", transform=trans)
        return [txt]
class Visualizer():
    """ Class that handles graph visualization """

    def __init__(self,graph,layout=networkx.spring_layout,title:str = None, legend:str = None):
        self.graph = graph
        self.active_nodes = [node for node,data in self.graph.nodes(data=True) if data['activated']]
        self.hosts = [node for node,data in self.graph.nodes(data=True) if data['host']]
        self.nodes_offline = [node for node,data in self.graph.nodes(data=True) if data['offline']]
        self.layout = layout
        self.title = title
        self.legend = legend
        self.figsize = (19.2*2, 10.8*2)
        self.node_color = 'skyblue'
        self.edge_color = 'black'
        self.node_size = 1000
        self.linewidths = 1
        self.font_size = 15
        """print('-----------------------------')
        print(f"nodes_offline:{len(self.nodes_offline)}")
        print(f"nodes_active:{len(self.active_nodes)}")
        print(f"nodes_hosting:{len(self.hosts)}")
        print(f"nodes_offline and hosting:{len([node for node in self.nodes_offline if node in self.hosts])}")
        print(f"nodes_offline and active:{len([node for node in self.nodes_offline if node in self.active_nodes])}")
        print('-----------------------------')"""
    
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
        only_host_nodes = {node for node in self.hosts if node not in self.active_nodes and node not in self.nodes_offline}
        if len(only_host_nodes) > 0 and len(only_host_nodes) < len(list(self.graph)):
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=only_host_nodes, node_color='red', node_size=self.node_size)
        only_active_nodes = {node for node in self.active_nodes if node not in self.hosts}
        if len(only_active_nodes) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=only_active_nodes, node_color='green', node_size=self.node_size)
        combination_nodes = {node for node in self.active_nodes if node in self.hosts}
        if len(combination_nodes) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=combination_nodes, node_color='orange', node_size=self.node_size)
        if len(self.nodes_offline) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=self.nodes_offline, node_color='grey', node_size=self.node_size)
        edge_labels = {(u, v): f"{round(d['time'],2)}" for u, v, d in self.graph.edges(data=True) if d['time'] > 0}
        networkx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        if self.legend:
            legend_elements = [plt.Line2D([0], [0], linestyle="none")]  # Placeholder for the custom handler
            plt.legend(handles=legend_elements, bbox_to_anchor=(0.88, 0.02), loc='lower right', handler_map={legend_elements[0]: TextLegendHandler(self.legend, fontsize=20, )})
            plt.subplots_adjust(right=0.75)
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
        only_host_nodes = {node for node in self.hosts if node not in self.active_nodes and node not in self.nodes_offline}
        if len(only_host_nodes) > 0 and len(only_host_nodes) < len(list(self.graph)):
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=only_host_nodes, node_color='red', node_size=self.node_size)
        only_active_nodes = {node for node in self.active_nodes if node not in self.hosts}
        if len(only_active_nodes) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=only_active_nodes, node_color='green', node_size=self.node_size)
        combination_nodes = {node for node in self.active_nodes if node in self.hosts}
        if len(combination_nodes) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=combination_nodes, node_color='orange', node_size=self.node_size)
        if len(self.nodes_offline) > 0:
            networkx.draw_networkx_nodes(self.graph, pos, nodelist=self.nodes_offline, node_color='grey', node_size=self.node_size)
        edge_labels = {(u, v): f"{round(d['time'],2)}" for u, v, d in self.graph.edges(data=True) if d['time'] > 0}
        networkx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        if self.legend:
            legend_elements = [plt.Line2D([0], [0], linestyle="none")]  # Placeholder for the custom handler
            plt.legend(handles=legend_elements, bbox_to_anchor=(0.88, 0.02), loc='lower right', handler_map={legend_elements[0]: TextLegendHandler(self.legend, fontsize=20, )})
            plt.subplots_adjust(right=0.75)
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
    
def paint_chart(x:list,y:list,label_x:str,label_y:str,title:str,filepath:str,**kwargs):
    plt.figure(figsize=(19.2, 10.8))
    plt.plot(x, y, **kwargs)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.savefig(filepath)
    plt.close()