import numba
import random
import math
import copy # for array copying
import sys
import numpy as np
from numba import jitclass, jit
from numba import int32, float32, int64

spec = [
    ('num_dimensions', int32),
    ('position_i', float32[:]),         # particle position
    ('velocity_i', float32[:]),
    ('pos_best_i', float32[:]),
    ('err_best_i', float32),
    ('err_i', float32)        
]

@jitclass(spec)
class Particle:
    def __init__(self, x0, num_dimensions):
        
        self.num_dimensions = num_dimensions
        self.position_i = np.zeros((1,self.num_dimensions), dtype=np.float32)          # particle position
        self.velocity_i = np.zeros((1,self.num_dimensions), dtype=np.float32)          # particle velocity
        self.pos_best_i = np.zeros((1,self.num_dimensions), dtype=np.float32)         # best position individual
        self.err_best_i = -1          # best error individual
        self.err_i = -1               # error individual
        
        for i in range(0, self.num_dimensions):
            self.velocity_i[i] = random.uniform(-1,1)
            self.position_i[i] = x0[i]

    # evaluate current fitness
    def evaluate(self):
        
        self.err_i = error(self.position_i)
        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1        # cognative constant
        c2 = 2        # social constant

        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, self.num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


@jit(parallel=True)
def error(position):
    err = 0.0
    for i in range(len(position)):

        xi = position[0][i]
        err += (xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10
    return err