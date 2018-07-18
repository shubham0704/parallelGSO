import math
import numpy as np
from numba import jit
from failed_exec import create_n_particles
from evaluate import evaluate, update_velocity, update_position

@jit
def error(position):
    err = 0.0
    for i in range(len(position)):
    
        xi = position[0][i]
        err += (xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10
    return err

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

    return pos_best_g, err_best_g

if __name__ == '__main__':
    num_particles = 15
    swarm_init = [np.random.uniform(-10,10, 2) for _ in range(num_particles)]
    bounds=[(-10,10),(-10,10)]
    PSO(error,bounds,maxiter=3, swarm_init=swarm_init)
    print('working')
#     particles = create_n_particles(100, 2, bounds)
#     #evaluate(particles[0])
#     pos_best_g = particles[0]['position_i']
#     update_velocity(pos_best_g, particles[1])
#     update_position(bounds, particles[1])
    