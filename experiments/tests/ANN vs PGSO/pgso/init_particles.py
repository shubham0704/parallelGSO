from numba import jit
import numpy as np


@jit
def create_n_particles(n, num_dimensions, swarm_init):
    
    particle_dtypes = np.dtype({'names':['position_i', 'velocity_i', 'pos_best_i',
                                      'err_best_i', 'err_i', 'num_dimensions'],
                                'formats':[(np.double,(1,(num_dimensions))),
                                       (np.double,(1,(num_dimensions))),
                                       (np.double,(1,(num_dimensions))),
                                       np.double, np.double, np.int32]
                            })
    particles = np.empty(n, dtype=particle_dtypes)
    for p,x0 in zip(particles, swarm_init):
        p['err_best_i'] = -1
        p['err_i'] = -1
        p['num_dimensions'] = num_dimensions
        for i in range(num_dimensions):
            p['velocity_i'][0][i] = np.random.uniform(-1,1)
            p['position_i'][0][i] = x0[i]
    return particles

if __name__ == '__main__':
    #swarm_init = [(5, 5) for _ in range(100)]
    num_particles = 15
    swarm_init = [np.random.uniform(-10,10, 2) for _ in range(num_particles)]
    particles = create_n_particles(100, 2, swarm_init)
    print(particles[:3])