from evaluate import error, evaluate, update_velocity, update_position
from failed_exec import create_n_particles
from pso_fail import PSO
from evaluate import error
from numba import jit, njit
import numba
import numpy as np
M = [5, 10, 15, 20, 25, 30, 35, 40]
bounds = [[-10, 10], [-10, 10]]
num_particles = 35
max_iter = 30
m = 5

@jit
def GSO(M, bounds, num_particles, max_iter):
    subswarm_bests = []
    dims = len(bounds)
    lb = bounds[0][0] 
    ub = bounds[0][1] 
    for i in range(M):
        #initial= np.random.uniform(-10,10, 2)               # initial starting location [x1,x2...]         
        swarm_init = []
        for _ in range(num_particles):
            swarm_init.append(np.random.uniform(lb, ub, dims))
        subswarm_best,_ = PSO(error,bounds,max_iter, swarm_init=swarm_init)
        subswarm_bests.append(subswarm_best[0])
    #return subswarm_bests
    return PSO(error, bounds, max_iter, swarm_init=subswarm_bests)

GSO(m, bounds, num_particles, max_iter)