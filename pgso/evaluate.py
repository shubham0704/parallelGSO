from pgso.test_functions import *
from pgso.init_particles import create_n_particles
from numba import jit
import math
import random

@jit(debug=True)
def error(position):
    err = 0.0
    for i in range(len(position)):
    
        xi = position[0][i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return err

#@jit
def evaluate(costFunc, p):
    
    try:
        p['err_i'] = costFunc(p['position_i'][0])
    except:
        p['err_i'] = costFunc(p['position_i'])

    # check to see if the current position is an individual best
    if p['err_i'] < p['err_best_i'] or p['err_best_i']==-1:
        p['pos_best_i'] = p['position_i']
        p['err_best_i'] = p['err_i']
    return p['pos_best_i'], p['err_best_i']

@jit
def update_velocity(pos_best_g, p):
    w=0.5       # constant inertia weight (how much to weigh the previous velocity)
    c1=1        # cognative constant
    c2=2        # social constant

    for i in range(0, p['num_dimensions']):
        r1=random.random()
        r2=random.random()

        vel_cognitive=c1*r1*(p['pos_best_i'][0][i]-p['position_i'][0][i])
        vel_social=c2*r2*(pos_best_g[0][i]-p['position_i'][0][i])
        p['velocity_i'][0][i]=w*p['velocity_i'][0][i]+vel_cognitive+vel_social
    return p['velocity_i']

@jit
def update_position(bounds, p):
    for i in range(0, p['num_dimensions']):
        p['position_i'][0][i]=p['position_i'][0][i]+p['velocity_i'][0][i]

        # adjust maximum position if necessary
        if p['position_i'][0][i]>bounds[i][1]:
            p['position_i'][0][i]=bounds[i][1]

        # adjust minimum position if neseccary
        if p['position_i'][0][i] < bounds[i][0]:
            p['position_i'][0][i]=bounds[i][0]
    return p['position_i']
