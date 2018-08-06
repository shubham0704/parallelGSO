# parallelGSO

<center><image src="images/cover_pso.png"><center>
    
A highly scalable parallel version of Galactic Swarm Optimisation Algorithm

Galactic Swarm Optimization is a state-of-the-art meta-heuristic optimization algorithm which is insiped by the motion of stars, galaxies, superclusters interacting with each other under the influence of gravity.

----
Take a look at a detailed introduction to our project - **[HERE](https://github.com/shubham0704/parallelGSO/blob/master/white%20paper.ipynb)**

-------

# Installation

Anaconda Users - 
```
$ conda env update -f env.yaml
$ conda activate pgso
```

PIP Users - 
```
pip install -r requirements.txt
```


# To run Benchmarks

All the testing experiments are present in the experiments/tests directory.
To rerun the benchmarks do -
```
// cd into this project directory then
$ cd experiments/tests
$ jupyter notebook
```

You will then find a lot of notebooks which contains all kinds of different testings
To run the main experiments, check the [Main Experiments (Performance Tests) notebook](https://github.com/shubham0704/parallelGSO/blob/master/experiments/tests/Main%20Experiments%20(Performance%20Tests).ipynb) 



**To Use PGSO (Parallel Galactic Swarm Optimization) as a module do -** 

```
from pgso.gso import GSO as PGSO
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

**To test using our provided test functions**

```
available_functions = [sphere, rosen, rastrigin, griewank, zakharov, nonContinuousRastrigin]
from pgso.test_functions import <function>

