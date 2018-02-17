from fail_exec import create_n_particles
from numba import jit
import math


particles = create_n_particles(100, 2, bounds)

@jit
def error(position):
  err = 0.0
  for i in range(len(position)):
    xi = position[i]
    err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
  return err

@jit
def evaluate(p):
    p['err_i'] = error(p['position_i'])

    # check to see if the current position is an individual best
    if p['err_i'] < p['err_best_i'] or p['err_best_i']==-1:
        p['pos_best_i'] = p['position_i']
        p['err_best_i'] = p['err_i']
    return [p['pos_best_i'], p['err_best_i']]

ans = evaluate(particles[0])