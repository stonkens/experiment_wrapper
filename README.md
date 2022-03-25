# Experiment Wrapper
Wrapper for experiments that do not have gym interface. Interfaces with custom-defined dynamics and controllers / policies. Facilitates parallelizing experiments either during training validation or testing. Inspired by [Neural CLBF](https://github.com/MIT-REALM/neural_clbf) package from MIT REALM. 

## Installation
The package and its dependencies can be installed from `pypi` using `pip install experiment-wrapper` [link](https://pypi.org/project/experiment-wrapper/).

Users can also fork this repository and clone it locally and install requirements from `requirements.txt` file or using the `pip install -e .` command.

## Instructions
The `Experiment` class and `ExperimentSuite` class provide the generic API to set up any type of experiment. An experiment is anything that tests the behavior / performance of a controller / policy. The `Experiment` class is used to set up one experiment and run / save / plot it. The `ExperimentSuite` class is used to run multiple experiments in parallel.

A single experiment consists of running multiple policies, from multiple different initial conditions, under multiple scenarios (e.g. different instantiations of the uncertain parameters), with multiple rollouts (e.g. for stochastic policies / environments / measurements / dynamics).

Running an experiment requires having a `Dynamics` class and a callable `Controller` (class or function) that takes a state and returns an action. Templates for these two classes are provided in the `experiment_wrapper/__init__.py` file.

## Use cases

Possible types of experiments include: 
- Log the accumulated reward of a policy on different scenarios and different initial conditions.
- Log the failure rate of a policy on different scenarios and different initial conditions.
- Log the state space evolution of a policy on different scenarios and different initial conditions.

The `rollout_trajectory.py` file contains classes that log (at least) the states and controls of the dynamics and its policy over time. An example of its implementation using CBFs to solve an [Adaptive Cruise Control](https://github.com/stonkens/cbf_opt/blob/main/examples/acc.ipynb) problem is provided in the [`cbf_opt`](https://github.com/stonkens/cbf_opt) repository.

## TODOs
Aside from the specific TODOs scattered throughout the codebase, a few general TODOs:
- [ ] Test functionality with torch.Tensor()'s
- [ ] Add an example in this repo for clarification
- [ ] Think about the ExperimentSuite functionality!
