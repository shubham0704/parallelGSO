
from evaluate import error, evaluate, update_velocity, update_position
from failed_exec import create_n_particles
from numba import jit
import numpy as np
from multiprocessing import Manager, Process

@jit
def PSO(costFunc,bounds,maxiter,shared_list,num_particles=None,swarm_init=None):

    if num_particles is not None:
        dims = len(bounds)
        lb = bounds[0][0] 
        ub = bounds[0][1]
        swarm_init = []
        for _ in range(num_particles):
            swarm_init.append(np.random.uniform(lb, ub, dims))
        
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
        
        # update the global best in the manager list after k iterations
        # we need to add some mutex lock here
        if i == max_iter//2:
            best_galactic_pos = shared_list[0]
            best_galactic_err = shared_list[1]
            if err_best_g < best_galactic_err:
                shared_list[1] = err_best_g
                shared_list[0] = pos_best_g
            else:
                err_best_g = best_galactic_err
                pos_best_g = best_galactic_pos
        
        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j]['velocity_i'] = update_velocity(pos_best_g, swarm[j])
            swarm[j]['position_i'] = update_position(bounds, swarm[j])
        i+=1

def start(process_list):
    for p in process_list:
        p.start()
        
def stop(process_list):
    result = []
    for p in process_list:
        result.append(p.join())
    return result

@jit
def GSO(M, bounds, num_particles, max_iter):
    subswarm_bests = []
    dims = len(bounds)
    lb = bounds[0][0] 
    ub = bounds[0][1]
    manager = Manager()
    shared_list = manager.list()
    shared_list = [np.random.uniform(lb, ub, dims), -1]
    
    all_processes = []
    for i in range(M):
        #initial= np.random.uniform(-10,10, 2)               # initial starting location [x1,x2...]         
        swarm_init = []
        for _ in range(num_particles):
            swarm_init.append(np.random.uniform(lb, ub, dims))
        p = Process(target=PSO, args=(error,bounds,max_iter,shared_list, swarm_init=swarm_init))
        all_processes.append(p)
    
    start(all_processes)
    subswarm_bests = stop(all_processes)    
        
    #subswarm_bests.append(subswarm_best[0])
    #return subswarm_bests
    return PSO(error, bounds, max_iter, swarm_init=subswarm_bests)