from evaluate import error, evaluate, update_velocity, update_position
from failed_exec import create_n_particles
from numba import jit
import numpy as np


@jit
def PSO(costFunc,bounds,maxiter, swarm_init):

    num_dimensions=len(swarm_init[0])
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group
    num_particles = len(swarm_init)
    # establish the swarm
    swarm = create_n_particles(num_particles, num_dimensions, swarm_init)
    # begin optimization loop
    i=0
    while i < maxiter:
        #print i,err_best_g
        # cycle through particles in swarm and evaluate fitness
        for j in range(0,num_particles):
            swarm[j]['pos_best_i'], swarm[j]['err_best_i']  = evaluate(costFunc, swarm[j])

            # determine if current particle is the best (globally)
            if swarm[j]['err_i'] < err_best_g or err_best_g == -1:
                pos_best_g=list(swarm[j]['position_i'])
                err_best_g=float(swarm[j]['err_i'])

        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j]['velocity_i'] = update_velocity(pos_best_g, swarm[j])
            swarm[j]['position_i'] = update_position(bounds, swarm[j])
        i+=1

    # print final results
    #print ('\n')
    #print (pos_best_g,' , ', err_best_g)
    return pos_best_g, err_best_g

bounds=[[-10,10],[-10,10]]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
num_particles = 15
swarm_init = [np.random.uniform(-10,10, 2) for _ in range(num_particles)]
PSO(error,bounds,maxiter=30, swarm_init=swarm_init)
#numba_time =  %%timeit -o PSO(error,bounds,maxiter=30, swarm_init=swarm_init)
