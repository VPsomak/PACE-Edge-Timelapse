from grapher import Grapher
import sys

model = None
graph_type = None

if len(sys.argv) == 3:
    model = sys.argv[1]
    graph_type = sys.argv[2]
else:
    print ("Wrong number of arguments!\n")
    print ("Example: python3 grapher.py [model] [algorithm] \n")
    print ("Available models [ilp, approximation, greedy, genetic] \n")
    print ("Available graphs [binomial_tree, balanced_tree, star, barabasi_albert, erdos_renyi, newman_watts_strogatz]")
    sys.exit()

grapher = Grapher(model = model,graph_type = graph_type)
grapher.solve()