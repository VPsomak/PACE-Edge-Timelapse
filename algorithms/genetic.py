import networkx as nx
import numpy as np
from random import randint, uniform, choice, shuffle

# a weighted choice function
def weighted_choice(choices, weights):
    normalized_weights = np.array([weight for weight in weights]) / np.sum(weights)
    threshold = uniform(0, 1)
    total = 1
    for index, normalized_weight in enumerate(normalized_weights):
        total -= normalized_weight
        if total < threshold:
            return choices[index]
        
class Population:
    """ Representation of a population of vertex covers """
    def __init__(self, G:nx.Graph, population_size:int, volume:int, elite_population_size:int,mutation_probability:float):
        self.vertexcovers:list[VertexCover] = []
        self.population_size = population_size
        self.graph = G.copy()
        self.nodes_activated = [node for node,data in self.graph.nodes(data=True) if data['activated']]
        self.old_hosts = [node for node,data in self.graph.nodes(data=True) if data['host']]
        self.imageSize = volume
        self.elite_population_size = elite_population_size
        self.mutation_probability = mutation_probability

        for vertex_cover_number in range(self.population_size):
            vertex_cover = VertexCover(self)
            vertex_cover.evaluate_fitness()

            self.vertexcovers.append(vertex_cover)
            self.vertexcovers[vertex_cover_number].index = vertex_cover_number

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.mean_fitness = 0
        self.mean_diversity = 0
        self.mean_vertex_cover_size = 0
        self.average_vertices = np.zeros((len(self.graph.nodes()), 1))

    # evaluate fitness ranks for each vertex cover
    def evaluate_fitness_ranks(self):
        if not self.evaluated_fitness_ranks:
            for vertex_cover in self.vertexcovers:
                vertex_cover.fitness = vertex_cover.evaluate_fitness()
                self.mean_fitness += vertex_cover.fitness
                self.mean_vertex_cover_size += len(vertex_cover)

            self.mean_fitness /= self.population_size
            self.mean_vertex_cover_size /= self.population_size
            self.vertexcovers.sort(key=lambda vertex_cover: vertex_cover.fitness, reverse=True)

            for rank_number in range(self.population_size):
                self.vertexcovers[rank_number].fitness_rank = rank_number

            self.evaluated_fitness_ranks = True

    # evaluate diversity rank of each point in population
    def evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            # find the average occurrence of every vertex in the population
            for vertex_cover in self.vertexcovers:
                self.average_vertices[vertex_cover.vertexlist] += 1

            self.average_vertices /= self.population_size

            for vertex_cover in self.vertexcovers:
                vertex_cover.diversity = np.sum(np.abs(vertex_cover.vertexlist - self.average_vertices))/self.population_size
                self.mean_diversity += vertex_cover.diversity

            self.mean_diversity /= self.population_size
            self.vertexcovers.sort(key=lambda vertex_cover: vertex_cover.diversity, reverse=True)

            for rank_number in range(self.population_size):
                self.vertexcovers[rank_number].diversity_rank = rank_number

            self.evaluated_diversity_ranks = True

    # generate the new population by breeding vertex covers
    def breed(self):
        # sort according to fitness_rank
        self.vertexcovers.sort(key=lambda vertex_cover: vertex_cover.fitness_rank)

        # push all the really good ('elite') vertex covers first
        newpopulation = []
        for index in range(self.elite_population_size):
            newpopulation.append(self.vertexcovers[index])

        # assign weights to being selected for breeding
        weights = [1 / (1 + vertex_cover.fitness_rank + vertex_cover.diversity_rank) for vertex_cover in self.vertexcovers]

        # randomly select for the rest and breed
        while len(newpopulation) < self.population_size:
            parent1 = weighted_choice(list(range(self.population_size)), weights)
            parent2 = weighted_choice(list(range(self.population_size)), weights)

            # don't breed with yourself, dude!
            while parent1 == parent2:
                parent1 = weighted_choice(list(range(self.population_size)), weights)
                parent2 = weighted_choice(list(range(self.population_size)), weights)

            # breed now
            child1, child2 = self.vertexcovers[parent1].crossover(self.vertexcovers[parent2])

            # add the children
            newpopulation.append(child1)
            newpopulation.append(child2)

        # assign the new population
        self.vertexcovers = newpopulation

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False

    # mutate population randomly
    def mutate(self):
        for vertex_cover in self.vertexcovers:
            test_probability = uniform(0, 1)
            if test_probability < self.mutation_probability:
                vertex_cover.mutate()
                vertex_cover.evaluate_fitness()

                self.evaluated_fitness_ranks = False
                self.evaluated_diversity_ranks = False

class VertexCover:
    """ Representation of a single vertex cover solution """
    def __init__(self, associated_population:Population=None):
        # population to which this point belongs
        self.associated_population = associated_population

        # randomly create chromosome
        self.chromosomes = [randint(0, 1) for _ in range(len(self.associated_population.graph.nodes()))]

        # initialize
        self.vertexarray = np.array([False for _ in range(len(self.associated_population.graph.nodes()))])
        self.chromosomenumber = 0
        self.vertexlist = np.array([])
        self.transfered = {edge:0 for edge in self.associated_population.graph.edges}

        # required when considering the entire population
        self.index = -1
        self.fitness = 0.0
        self.diversity = 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1
        self.evaluated_fitness = False

    def evaluate_fitness(self):
        if not self.evaluated_fitness:
            original_graph = self.associated_population.graph.copy()
            imageSize = self.associated_population.imageSize

            self.vertexarray = np.array([False for _ in range(len(original_graph.nodes()))])
            self.transfered = {edge:0 for edge in self.associated_population.graph.edges}
            self.chromosomenumber = 0

            while len(original_graph.edges) > 0:
                # shuffle the list of vertices
                node_list = list(original_graph.nodes)
                shuffle(node_list)

                # remove all degree-1 vertices one-by-one
                degree_one_vertex_found = False
                for vertex in node_list:
                    if original_graph.degree[vertex] == 1 and original_graph.degree[list(original_graph.neighbors(vertex))[0]] == 1:
                        degree_one_vertex_found = True

                        neighbors = list(original_graph.neighbors(vertex))
                        adjvertex = neighbors[0]

                        # select the adjacent vertex
                        if adjvertex in self.associated_population.nodes_activated:
                            self.vertexarray[adjvertex] = True

                            if vertex in self.associated_population.nodes_activated:
                                # add a transfer on the edge just used
                                if (vertex,adjvertex) in self.transfered:
                                    self.transfered[(vertex,adjvertex)] += imageSize
                                else:
                                    self.transfered[(adjvertex,vertex)] += imageSize

                        # remove vertex along with neighbours from graph
                        removed_subgraph = neighbors
                        removed_subgraph.append(vertex)
                        original_graph.remove_nodes_from(removed_subgraph)
                        break

                # no more degree-1 vertices left
                if not degree_one_vertex_found:                    
                    # randomly choose one of the remaining vertices
                    vertex = choice(list(original_graph.nodes))
                    if original_graph.degree[vertex] >= 2:
                        # make a choice depending on the chromosome bit
                        if self.chromosomes[self.chromosomenumber] == 0:
                            # add all neighbours to vertex cover
                            activated_neighbours = []
                            for othervertex in original_graph.neighbors(vertex):
                                self.vertexarray[othervertex] = True
                                if vertex in activated_neighbours:
                                    if (vertex,othervertex) in self.transfered:
                                        self.transfered[(vertex,othervertex)] += imageSize
                                    else:
                                        self.transfered[(othervertex,vertex)] += imageSize

                            # remove vertex along with neighbours from graph
                            removed_subgraph = list(original_graph.neighbors(vertex))
                            removed_subgraph.append(vertex)
                            original_graph.remove_nodes_from(removed_subgraph)

                        elif self.chromosomes[self.chromosomenumber] == 1:
                            # add only this vertex to the vertex cover
                            if vertex in self.associated_population.nodes_activated:
                                self.vertexarray[vertex] = True
                                # remove only this vertex from the graph
                            original_graph.remove_node(vertex)

                        # go to the next chromosome to be checked
                        self.chromosomenumber = self.chromosomenumber + 1
                        continue
                    

            # put all true elements in a numpy array - representing the actual vertex cover
            self.vertexlist = np.where(self.vertexarray == True)[0]

            request_failed_nodes = []
            image_placement_cost = self.associated_population.imageSize / 524288
            
            for vertex in self.associated_population.nodes_activated:
                if not vertex in self.vertexlist:
                    gotImage = False
                    for edge in self.transfered:
                        if (edge[0] == vertex or edge[1] == vertex) and self.transfered[edge] > 0:
                            gotImage = True
                    for other_vertex in self.vertexlist:
                        if not gotImage:
                            if (vertex,other_vertex) in self.transfered:
                                self.transfered[(vertex,other_vertex)] += imageSize
                                gotImage = True
                            elif (other_vertex,vertex) in self.transfered:
                                gotImage = True
                                self.transfered[(other_vertex,vertex)] += imageSize
                    if not gotImage:
                        request_failed_nodes.append(vertex)
            if len(request_failed_nodes) > 0:
                self.fitness = 0.0
            else:
                preserved_hosts = [vertex for vertex in self.vertexlist if vertex in self.associated_population.old_hosts]
                newhosts = [vertex for vertex in self.vertexlist if vertex not in self.associated_population.old_hosts]
                hostonly = [vertex for vertex in newhosts if vertex not in self.associated_population.nodes_activated]
                mixed = [vertex for vertex in newhosts if vertex in self.associated_population.nodes_activated]
                total_transfer_time = sum([self.transfered[edge]/self.associated_population.graph.edges[edge[0],edge[1]]['capacity'] for edge in self.associated_population.graph.edges])
                unused_hosts = []
                for host in hostonly:
                    host_edges = self.associated_population.graph.edges(host)
                    if sum([self.transfered.get(host_edge,0) for host_edge in host_edges]) == 0:
                        unused_hosts.append(host)
                score = total_transfer_time + \
                    (len(unused_hosts) * (image_placement_cost * 4)) + \
                    (len(mixed) * (image_placement_cost)) + \
                    (len(preserved_hosts) * (image_placement_cost / 4))
                if score == 0:
                    self.fitness = 1.0
                else:
                    self.fitness = 1000 / score
            self.evaluated_fitness = True
        
        return self.fitness

    def mutate(self):
        """ Mutates the chromosome at a random index """
        if self.chromosomenumber > 0:
            index = randint(0, self.chromosomenumber)
        else:
            index = 0

        if self.chromosomes[index] == 0:
            self.chromosomes[index] = 1
        elif self.chromosomes[index] == 1:
            self.chromosomes[index] = 0

        self.evaluated_fitness = False
        self.evaluate_fitness()

    def __len__(self):
        return len(self.vertexlist)

    def __iter__(self):
        return iter(self.vertexlist)
    
    def crossover(self, parent2) -> tuple:
        """ Crossover between two vertex cover chromosomes """
        if self.associated_population != parent2.associated_population:
            raise ValueError("Vertex covers belong to different populations.")
        child1 = VertexCover(self.associated_population)
        child2 = VertexCover(parent2.associated_population)
        # find the point to split and rejoin the chromosomes
        # note that chromosome number + 1 is the actual length of the chromosome in each vertex cover encoding
        split_point = randint(0, min(self.chromosomenumber, parent2.chromosomenumber))
        # actual splitting and recombining
        child1.chromosomes = self.chromosomes[:split_point] + parent2.chromosomes[split_point:]
        child2.chromosomes = parent2.chromosomes[:split_point] + self.chromosomes[split_point:]
        # evaluate fitnesses
        child1.evaluate_fitness()
        child2.evaluate_fitness()
        return (child1, child2)

class GeneticSolver():
    """ The solver class using genetic algorithm """

    def __init__(self,graph:nx.Graph,image_size:int):
        # Constants
        # note: keep population_size and elite_population_size of same parity
        self.population_size = 20
        self.elite_population_size = 4
        self.mutation_probability = 0.04
        self.num_iterations = 5

        # Parameters
        self.graph=graph
        self.image_size = image_size

        # Initialise
        self.population = Population(
            self.graph, 
            self.population_size, 
            self.image_size, 
            self.elite_population_size, 
            self.mutation_probability
        )
        self.population.evaluate_fitness_ranks()
        self.population.evaluate_diversity_ranks()
        self.coverset = []

    def solve(self):
        """ Runs the solver """
        self.coverset = []

        # for plotting
        #plot_fitness = [self.population.mean_fitness]
        #plot_diversity = [self.population.mean_diversity]

        # breed and mutate this population num_iterations times
        for _ in range(1, self.num_iterations + 1):
            self.population.breed()
            self.population.mutate()
            # find the new ranks
            self.population.evaluate_fitness_ranks()
            self.population.evaluate_diversity_ranks()
            # add to the plot
            #plot_fitness.append(self.population.mean_fitness)
            #plot_diversity.append(self.population.mean_diversity)

        # vertex cover with best fitness is our output
        best_vertex_cover = None
        best_fitness = 0
        for vertex_cover in self.population.vertexcovers:
            if vertex_cover.fitness > best_fitness:
                best_vertex_cover = vertex_cover
                best_fitness = vertex_cover.fitness
        if best_vertex_cover is not None:
            list_mvc = best_vertex_cover.vertexlist.tolist()
            self.coverset.append(list_mvc)
            self.coverset.append(best_fitness)
            print(f"Best Fitness {best_fitness}")
            self.coverset.append(best_vertex_cover.transfered)
        else:
            self.coverset.append([])
            self.coverset.append(-1)
            print(f"Best Fitness {-1}")
            self.coverset.append([])
