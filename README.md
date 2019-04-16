# parallelGSO

<center><image src="images/cover_pso.png"><center>
    
A highly scalable parallel version of Galactic Swarm Optimisation Algorithm

Galactic Swarm Optimization is a state-of-the-art meta-heuristic optimization algorithm which is insiped by the motion of stars, galaxies, superclusters interacting with each other under the influence of gravity.

Train Artificial Neural Networks quickly without backprop!

----
Take a look at a detailed introduction to our project - **[HERE](https://github.com/shubham0704/parallelGSO/blob/master/white%20paper.ipynb)**

-------

# Installation

We recommend Anaconda. For installing Anaconda (for Linux) you can use this [script](https://github.com/shubham0704/deep_learning_env/blob/master/install_anaconda.sh)

For Anaconda Users - 
```
$ conda env update -f env.yaml
$ conda activate pgso
```
For PIP Users - 
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

To run the main experiments, check -[Main Experiments (Performance Tests)](https://github.com/shubham0704/parallelGSO/blob/master/experiments/tests/Main%20Experiments%20(Performance%20Tests).ipynb) 

For per-cpu utilization benchmarks check - [Per CPU Utilisation Experiments](https://github.com/shubham0704/parallelGSO/blob/master/experiments/tests/Per%20CPU%20Utilisation%20Experiments.ipynb)

To run Benchmarks against the functions test suite - [Benchmarks](https://github.com/shubham0704/parallelGSO/blob/master/experiments/Benchmarks.ipynb)

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

**Training Artificial Neural Networks**

Check out [ANN training](https://github.com/shubham0704/parallelGSO/tree/master/experiments/tests/ANN%20vs%20PGSO)
This directory contains tutorial notebooks for you to get started.
