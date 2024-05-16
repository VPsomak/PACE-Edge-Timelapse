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

from pace import PACE
import sys
import random
from math import floor, ceil

#TODO: Add image relocation to score ( relocating an image is more costly than keeping an image in place )

model = None
graph_type = None

### OPTIONS ###
random.seed(10)
hw_fault_propability = 0.015
activation_start_ratio = 0.2
activation_ratio_step = 0.05
timesteps = 40
### END OPTIONS ###

if len(sys.argv) == 3:
    model = sys.argv[1]
    graph_type = sys.argv[2]
else:
    print ("Wrong number of arguments!\n")
    print ("Example: python3 main.py [model] [algorithm] \n")
    print ("Available models [ilp, approximation, greedy, genetic] \n")
    print ("Available graphs [binomial_tree, balanced_tree, star, barabasi_albert, erdos_renyi, newman_watts_strogatz]")
    sys.exit()

prev_graph = None
prev_ratio = activation_start_ratio
pos = None
for step in range(floor(timesteps/2.0)):
    pace = PACE(
        model = model,
        graph_type = graph_type,
        activated_ratio = prev_ratio,
        name=f"pace_{step}",
        seed=step,pos = pos,
        graph=prev_graph,
        hw_fault_probability=hw_fault_propability
    )
    pace.solve()
    print(f"\n{pace.solution_text}\n")
    pos = pace.pos
    prev_graph = pace.graph
    prev_ratio += activation_ratio_step
    prev_ratio = min(prev_ratio,1.0)

for step in range(floor(timesteps/2.0),timesteps):
    pace = PACE(
        model = model,
        graph_type = graph_type,
        activated_ratio = prev_ratio,
        name=f"pace_{step}",
        seed=step,
        pos = pos,
        graph=prev_graph,
        hw_fault_probability=hw_fault_propability
    )
    pace.solve()
    print(f"\n{pace.solution_text}\n")
    pos = pace.pos
    prev_graph = pace.graph
    prev_ratio -= activation_ratio_step
    prev_ratio = max(prev_ratio,activation_start_ratio)