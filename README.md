# Pro-active Component Image Placement in Edge Computing Environments (PACE-Edge)
  
We model the problem of proactive placement of application images in Edge computing as a Minimum Vertex Cover (MVC) problem. 

## Set Cover Problem

In this section, the proposed approach for the set cover problem is presented. Specifically, we first present the problem formulation, the constraints, and the optimization goal of the problem. Then, it demonstrates the solution space, thus presenting the data placement architecture. Finally, itintroduces the transformation of set cover to an optimization problem.

**Problem formulation**. Locate the nodes on which one or more replicas of the data objects must be placed in order to achieve the constraint for all nodes $d$, while minimizing the cost function for the whole network.

Constraint. $V_{obj} / W_{sd} < K$ 
Optimization Goal. $min(F(K,Vobj,Rs))$ 

$R$ and $W$ are considered as constants for each specific time window $t_0+K$, where $t_0$ is the starting time of our time window. That means that we are taking a snapshot of the network’s state at time $t_0$, and this snapshot has a lifetime of $K$. We are considering the available bandwidth $W_{sd}$ as equal to $W_{ds}$, meaning that the direction of the edge is not affecting its available bandwidth. To formulate the feasible solution space, we are considering all combinations of data placement on nodes s, having one or more replicas of the data object, and pruning all combinations that fail to achieve the constraint set by $K$ for each node $d$. Each such combination is considered a feasible solution. As optimal solutions, we consider the ones that are feasible and minimize the cost function $F$ for all nodes involved in them. Main notations are summarized below.

  **Notation & Description**

  Node $s$ & Source node of a data object  

  Node $d$ & Destination node of a data object 

  $V_{obj}$ & Volume of data object 

  $R_s$   & Charge rate for data storage in node $s$ 

  $W_{sd}$ & Available link bandwidth between nodes $s$ and $d$ 

  $K$      & Time constraint for data object transfer $\forall$ node $d$

  $F$      & Cost function of data object placement on node $s$ 

  $N$      & Set of all nodes in the network


**Solution Space**
A solution is in essence a data placement architecture. It describes which nodes will be used as data storage nodes and how many replicas of the data object will be placed on the network. As a result, each solution ($i$) is composed of two parts; a) the data placement architecture ($DP_i$) and b) the total cost ($TC_i$). The cost is derived using a function of the volume of the data object ($V_{obj}$), the cost rate of data storage in node $s$ ($R_s$), and the available storage capacity of node $s$ ($C_s$) for each node $s$ that is part of $DP_i$. For a solution to be considered feasible, it must preserve the constraint of data transfer time for all possible destination nodes $d$. This means that the time $V_{obj} / W_{sd}$, where $V_{obj}$ is the size (volume) of the object and $W_{sd}$ the available bandwidth of the connection, must be always less than the transfer time threshold $K$ for every node $d$ in the network $N$ and for at least one node $s$ that takes part in $DP_i$. 

This is described in a mathematical way in function:
$V_{obj} / W_{sd} < K \mid \exists s \in DP_i, \forall d \in N$

For the solution $i$ which is defined as:
$DP_i = [Node_0,Node_1,\dots,Node_s] \text{  and  }$
$TC_i = \sum_s{F(V_{obj},C_s,R_s) \mid \forall s \in DP_i}]$

**Optimization Problem**
We formalize the problem we want to solve as an optimization problem, i.e. in terms of objective function and constraints. In order to achieve that, the objective function and the restraints need to be adjusted in a form more fitting to linear optimization logic. The objective function can be considered a function of only the activated nodes and the volume of the image being transferred because, as discussed, all the other variables ($R,W$) are considered constants for the duration $K$ we are examining. The new objective function, which is also our minimization target is described as follows:

$min(\sum_{n}(A_{n} * V)) \mid A_{n} \in \{0,1\}, \forall n \in N$


This new objective function employs a new set of variables $A_{n}$ which are binary, taking the values of 1 or 0, describing the activated or deactivated state of a node $n$ respectively. In our case, activation means that node $n$ is holding a replica of the image and is able to distribute it to other nodes that are connected to it. 
The next step in defining a linear optimization problem is defining the set of restraints that need to be applied which are the following:

$\exists n \in N: A_{n} * V / W_{nd} < K \mid \exists d \in D$


This equation describes the restraint posed by the transfer time threshold. It states that if the optimization algorithm decides to activate a node $n$, it must ensure that the transfer time of the image from that node, given by the division of the image's volume $V$ and the available bandwidth $W_{nd}$ between the activated node $n$ and at least one destination node $d$ in the list of destination nodes $D$ that are requesting the image, must be under the threshold $K$. If the node is not activated the value of $A_{n}$ would be 0 so the restraint would hold true anyway. 

Trying to implement "at least one'' in a practical linear optimization algorithm is almost impossible. For this reason, we need to reformat this function, viewing it from different angles that encompass it. The rewritten format denotes that for each destination node $d$ the total transfer time of images from all nodes $n$ must be more than zero, ensuring that at least one of the nodes $n$ is activated and able to serve the image to the destination node $d$. This is described in the following function:

$\sum_{n}(A_{n} * V /  W_{nd}) > 0 \mid \forall n \in N, \forall d \in D$


Moreover, we need to ensure that the activated nodes are transferring the image following as close as possible to the time threshold $K$. This means that we have to relax the constraint of the time threshold, reducing it to a "should" rule instead of a "must" rule. This relaxation practically means that we expect some slight violations of the constraint to happen, translating into possible QoS violations in a real-life scenario. Since our target is the minimization of the function we can just include the minimization of total transfer time, which ensures that the transfer time will be kept as low as possible, covering, in most cases, our need for a transfer time lower than the threshold $K$. In order to mathematically depict this change we have to reformat our objective function, having the transfer time included in it as follows:

$min(\sum_{n}(A_{n} * V) + \sum_{n}\frac{A_{n} * V}{W_{nd}}) \\ 
A_{n} \in \{0,1\}, \forall n \in N, \forall d \in D$


Now that the objective function has two factors in it, another problem arises; we need to balance these two factors in order to have an equal impact on the final value of the function.
This problem arises from the fact that transfer times are usually counted in milliseconds while volume is counted in megabytes or gigabytes, which means that a change in the first factor of the function would greatly affect the final value while a change in the second factor would have only a slight effect. For that reason, we decided to remove the volume value from the first factor, applying only the sum of $A_{n}$ variables as a weight on the total transfer time in the network. This means that if a solution achieves the same or less total transfer time while using fewer nodes as image sources then it will be preferred over the others. ***The final objective function is the following***:


$min(\sum_{n}A_{n} + \sum_{n}\frac{A_{n} * V}{W_{nd}}) \\ 
A_{n} \in \{0,1\}, \forall n \in N, \forall d \in D$

## Placement approaches & topologies & Metrics
**Placement approaches**:

 - Integer Linear Programming (ILP)
 - Approximation
 - Greedy
 - Genetic

**Network topologies**:

 - Star
 - Binomial
 - Balanced
 - Erdo-Renyi
 - Watts-Strogatz
 - Barabasi-Albert

**Evaluation metrics**:

 - **Execution time**: refers to the total amount of time each algorithm requires to produce a solution
 - **Approximation ratio**: it is defined as the ratio of the VCS produced by an algorithm over the VCS produced by the optimal solution. ILP algorithm is guaranteed to be the optimal solution in producing the minimum vertex cover for a given graph and, therefore, is utilized as the benchmark of correctness, by which the approximation ratio seeks to evaluate all other algorithms with. 
-  **Cost function**: is the same as the objective function 
- **Vertex cover set size**: the size of vertices in vertex cover

 
## Usage:  
  
```python3 grapher.py [model] [algorithm]```  

 - Available placement algorithms: [ilp, approximation, bruteforce,
   branchandbound, genetic]  
   
 - Available graph topologies: [binomial_tree,   balanced_tree, star, barabasi_albert, erdos_renyi,    newman_watts_strogatz]



  
```python3 grapher.py genetic barabasi_albert```