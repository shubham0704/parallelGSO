# Installation

```
pip install requirements.txt
```

# To run Benchmarks

In the home directory of the project
```
from cuda_code.final.benchmark import monitor, multiple_cpu_plot

:parameters:
monitor(target=<GSO,PSO>, 
        bounds=<[[-100, 100],[-100, 100]]>, 
        num_particles=<number of particles>,
        max_iter=<maximum number of iterations>,
        costfunc=<n dimensional cost function>
        )
:returns:
        cpu_percents -> list of cpu_percent/number of proc (debug feature) [list]
        time_at -> time for plotting cpu usage [list]
        top_prcnt -> [[1,2,3,4,5,6,7,8], ....] list of cpu usage for each core

:parameters:
multiple_cpu_plot(top_prcnt,
                  time_at,
                  zoom_range = [lower, upper] -> zoom from one value to another
                  step = float -> how much to move from lower to upper by
                  )
:returns:
        will display maximum 8 cpu graph
        not scaled for more than 8 cpu because of subplots
```

# Run PGSO (parallel global swarm optimization)

```
from cuda_code.final.monolithic import GSO as PGSO
PGSO(
    M=<number of processes to be spawned
    bounds=<[[-100, 100],[-100, 100]]>, 
    num_particles=<number of particles>,
    max_iter=<maximum number of iterations>,
    costfunc=<n dimensional cost function>
    )

:returns:
    best_postition -> 1d array of n positions ex: [x, y] or [x,y,z] etc.
    best_error -> the best minimized error for the given function
```

# To test using our provided test functions

```
available_functions = [sphere, rosen, rastrigin, griewank, zakharov, nonContinuousRastrigin]
from cuda_code.final.test_functions import <function>
```