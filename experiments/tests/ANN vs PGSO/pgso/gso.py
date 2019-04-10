from pgso.evaluate import error, evaluate, update_velocity, update_position
from multiprocessing import Manager, Process, Lock
from pgso.init_particles import create_n_particles
from sklearn.utils import shuffle
from tqdm import tqdm
from numba import jit
import numpy as np
import copy


def sample_data(X_train, y_train, batch_size, mini_batch_size):
    X_train, y_train = shuffle(X_train, y_train) 
    for i in range(0, mini_batch_size-batch_size+1,batch_size):
        yield X_train[i:i+batch_size], y_train[i:i+batch_size]


# @jit
def PSO_purana(classifier,bounds,maxiter,swarm_init=None, train_data=None):
        
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
            swarm[j]['pos_best_i'], swarm[j]['err_best_i']  = evaluate(classifier, swarm[j], train_data)

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

# @jit
def PSO(classifier, bounds, maxiter, shared_list, return_list, l, num_particles=None, swarm_init=None, pso_train_data=None):

    # create minibatches inside PSO        
    num_dimensions=len(swarm_init[0])
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group
    num_particles = len(swarm_init)
    #print('adress of classifier object is: ', id(classifier))
    # establish the swarm
    # initialize swarm population
    #print('len(swarm_init): ', len(swarm_init), 'shape of swarm_init[0]: ', swarm_init[0].shape, '\n')
    swarm = create_n_particles(num_particles, num_dimensions, swarm_init)
    # begin optimization loop
    i=0
    while i < maxiter:
        #print i,err_best_g
        # cycle through particles in swarm and evaluate fitness
        for j in range(0,num_particles):
            best_pos, swarm[j]['err_best_i'] = evaluate(classifier, swarm[j], pso_train_data)
            swarm[j]['pos_best_i'] = best_pos
            # determine if current particle is the best (globally)
            if swarm[j]['err_i'] < err_best_g or err_best_g == -1:
                pos_best_g=list(swarm[j]['position_i'])
                err_best_g=float(swarm[j]['err_i'])
        
        # update the global best in the manager list after k iterations
        # we need to add some mutex lock here
        
        if i == maxiter//2:
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
    #print('shape of swarm[0][position_i] is: ', swarm[0]['position_i'].shape)
    return_list.append((pos_best_g[0], swarm[:]['position_i']))


def start(process_list):
    for p in process_list:
        p.start()
        
def stop(process_list):
    for p in process_list:
        p.join()

# @jit
def GSO(bounds, num_particles, max_iter, classifier, train_data, epochs, batch_size, mini_batch_size=None):
    """
    Galactic Swarm Optimization:
    ----------------------------
    A meta-heuristic algorithm insipred by the interplay
    of stars, galaxies and superclusters under the influence
    of gravity.
    
    Input:
    ------
    M: integer 
    number of galaxies
    bounds: 
    bounds of the search space across each dimension
    [lower_bound, upper_bound] * dims
    We specify only lower_bound and upper_bound
    
    """
    subswarm_bests = []
    dims = sum([np.prod(np.array(layer['weights']).shape) for layer in classifier.layers.values()])
    print("total number of weights -", dims)
    lb = bounds[0]
    ub = bounds[1]
    # lets set bounds across all dims
    bounds = [[lb, ub]]*dims
    manager = Manager()
    l = Lock()
    shared_list = manager.list()
    return_list = manager.list()
    shared_list = [np.random.uniform(lb, ub, dims), np.inf]
    all_processes = []
    #pso_batch_size = train_data[0].shape[0]//M
    g_best_weights = None
    g_best_error = float("inf")
    classifiers = [copy.deepcopy(classifier) for _ in range(mini_batch_size//batch_size)]

    X_train, y_train = train_data
    if not mini_batch_size: mini_batch_size = X_train.shape[0]

    print('starting with gso_batch size - {}, mini_batch_size -{} '.format(batch_size, mini_batch_size))
    
    # create N particles here
    swarm_inits = []
    for j in range(mini_batch_size//batch_size):
        swarm_init = []
        for _ in range(num_particles):
            swarm_init.append(np.random.uniform(lb, ub, (1, dims)))
        swarm_inits.append(swarm_init)

    for i in tqdm(range(epochs)):
        all_processes = []
        sampler = sample_data(X_train, y_train, batch_size, mini_batch_size)
        for j in range(mini_batch_size//batch_size):    
            pso_train_data = next(sampler)
            
            #initial= np.random.uniform(-10,10, 2)               # initial starting location [x1,x2...]         
            # swarm_init = []
            # for _ in range(num_particles):
            #     swarm_init.append(np.random.uniform(lb, ub, dims))
            
            #pso_train_data = (data[0][k*batch_size:(k+1)*pso_batch_size], data[1][k*batch_size:(k+1)*pso_batch_size])
            
            # print('started batch :',i)
            # print('train_data length :', len(pso_train_data))
            #print('shape of swarm_inits[j][0]: ', swarm_inits[j][0].shape)
            swarm_init = np.array([item.reshape(dims, 1) for item  in swarm_inits[j]])
            p = Process(target=PSO, args=(classifiers[j], bounds, max_iter, shared_list, return_list, l, None,swarm_init, pso_train_data))
            all_processes.append(p)

        start(all_processes)
        stop(all_processes)    
        #print('elements of return list: ', return_list)
        main_swarm_init = [item[0] for item in return_list]
        #swarm_inits = [item[1] for item in return_list]
        swarm_inits = [main_swarm_init for item in return_list]
        best_weights, best_error = PSO_purana(classifier, bounds, max_iter, swarm_init=main_swarm_init, train_data=train_data)

        if best_error < g_best_error:
            g_best_error = best_error
            g_best_weights = best_weights
        print('completed epoch {} --------> loss_value: {}'.format(i, best_error)) 

    prev_index = 0
    for layer_id, layer in classifier.layers.items():
        num_elements = np.prod(layer['weights'].shape) # we can cache this and pass it down or store it as layer.num_elements
        new_weights = g_best_weights[prev_index:prev_index+num_elements]
        layer['weights'] = new_weights.reshape(layer['weights'].shape) # changing value midway can cause some error
        prev_index += num_elements

    return classifier


# bounds are across each dimension
'''
Suppose you have 3 dims x,y,x

For each dimension you have to specify a range
x -> [-1, 1]
y -> [-10, 10]
z -> [-5, 5]
Your bounds array will look like - 
bounds = [[-1, 1], [-10, 10], [-5, 5]]

dims = sum([np.prod(layer['weights'].shape) for layer in classifier.layers])
bounds[0][0]
'''