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

model = None
graph_type = None

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
prev_ratio = 0.2
pos = None
random.seed(10)
for step in range(20):
    pace = PACE(model = model,graph_type = graph_type,activated_ratio = prev_ratio,name=f"pace_{step}",seed=step,pos = pos,graph=prev_graph)
    pace.solve()
    print(f"\n{pace.solution_text}\n")
    pos = pace.pos
    prev_graph = pace.graph
    prev_ratio += 0.05
    prev_ratio = min(prev_ratio,1.0)

for step in range(20,40):
    pace = PACE(model = model,graph_type = graph_type,activated_ratio = prev_ratio,name=f"pace_{step}",seed=step,pos = pos,graph=prev_graph)
    pace.solve()
    print(f"\n{pace.solution_text}\n")
    pos = pace.pos
    prev_graph = pace.graph
    prev_ratio -= 0.05
    prev_ratio = max(prev_ratio,0.2)