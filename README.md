# Deep HyperNEAT: Open-ended evolution of deep neural networks
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Using DeepHyperNEAT
In order to run DHN in its current form, you need to create a python script detailing the task.

The script needs to contain: substrate input and output dimensions, substrate inputs, population size, population elitism, number of generations, and a task function.

Please see xor_study.py as an example. 

## Primary Modules
These modules are associated with the primary function of the DeepHyperNEAT (DHN) algorihtm.
### genome.py
Contains all functionality of the genome, a Compositional Pattern Producing Network (CPPN) and its mutation operators.
### phenomes.py
Contains multiple representations for feed-forward and recurrent neural networks for the CPPN and the Substrate.
### population.py
Contains all functionality and information of the populations used in DHN.
### activations.py
A library of activation functions that can be used for the CPPN and Substrate.
### reproduction.py
Contains all functionality needed for the reproductive behavior in DHN.
### species.py
Contains all functionality needed for speciation in DHN.
### stagnation.py
Contains all functionality needed for stagnation schemes used in speciation.
### decode.py
Contains all functionality needed to decode a given CPPN into a Substrate.

## Secondary Modules
These modules are intended for secondary functionality such as reporting evolutionary statistics, visualizing the CPPN and Substrate, and various utility functions used throughout the primary modules.
### reporters.py
Contains various functions for reporting evolutionary statistics during and after an evolutionary run.
### visualize.py
Contains functions for visualizing a CPPN or Substrate.
### util.py
Contains common functions and iterators used throughout DHN.