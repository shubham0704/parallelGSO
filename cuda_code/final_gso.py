from evaluate import error, evaluate, update_velocity, update_position
from failed_exec import create_n_particles
from numba import jit
import numpy as np
from multiprocessing import Manager, Process, Lock


@jit
def PSO_purana(costFunc,bounds,maxiter,swarm_init=None):

  
        
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
    return pos_best_g[0], err_best_g

@jit
def PSO(costFunc,bounds,maxiter,shared_list, return_list, l,num_particles=None,swarm_init=None):

    
#     if num_particles is not None:
#         dims = len(bounds)
#         lb = bounds[0][0] 
#         ub = bounds[0][1]
#         swarm_init = []
#         for _ in range(num_particles):
#             swarm_init.append(np.random.uniform(lb, ub, dims))
        
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
            l.acquire()
            best_galactic_pos = shared_list[0]
            best_galactic_err = shared_list[1]
            #print("best_galactic_err: " ,best_galactic_err)
            #print("best_galactic_pos: ", best_galactic_pos)
            if err_best_g < best_galactic_err:
                shared_list[1] = err_best_g
                #print(err_best_g)
                shared_list[0] = pos_best_g
            else:
                #print("changing pos_best_g from", pos_best_g, " to ", best_galactic_pos)
                #emp_list = []
                err_best_g = float(best_galactic_err)
                #emp_list.append(best_galactic_pos)
                pos_best_g = [best_galactic_pos]
            
            l.release()
        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j]['velocity_i'] = update_velocity(pos_best_g, swarm[j])
            swarm[j]['position_i'] = update_position(bounds, swarm[j])
        i+=1
    return_list.append(pos_best_g[0])


def start(process_list):
    for p in process_list:
        p.start()
        
def stop(process_list):
    for p in process_list:
        p.join()

@jit
def GSO(M, bounds, num_particles, max_iter):
    subswarm_bests = []
    dims = len(bounds)
    lb = bounds[0][0] 
    ub = bounds[0][1]
    manager = Manager()
    l = Lock()
    shared_list = manager.list()
    return_list = manager.list()
    shared_list = [np.random.uniform(lb, ub, dims), -1]
    all_processes = []
    for i in range(M):
        #initial= np.random.uniform(-10,10, 2)               # initial starting location [x1,x2...]         
        swarm_init = []
        for _ in range(num_particles):
            swarm_init.append(np.random.uniform(lb, ub, dims))

        p = Process(target=PSO, args=(error, bounds, max_iter, shared_list, return_list, l, None,swarm_init))
        all_processes.append(p)

    start(all_processes)
    stop(all_processes)    
    #print(return_list)
    return PSO_purana(error, bounds, max_iter, swarm_init=list(return_list))

M = [5, 10, 15, 20, 25, 30, 35, 40]
bounds = [[-10, 10], [-10, 10]]
num_particles = 35
max_iter = 30
m = 5

GSO(5, bounds, num_particles, max_iter)