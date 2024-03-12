import networkx as nx
import numpy as np
import random
import numpy.random

random.seed(10)
numpy.random.seed(10)
imageSize = 3*1024*1024*1024

def max_min_fairness(demands, capacity):
    capacity_remaining = capacity
    output = []

    for i, demand in enumerate(demands):
        share = capacity_remaining / (len(demands) - i)
        allocation = min(share, demand)

        if i == len(demands) - 1:
            allocation = max(share, capacity_remaining)

        output.append(allocation)
        capacity_remaining -= allocation

    return output

