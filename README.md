# Deep HyperNEAT: Open-ended evolution of deep neural networks
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)



## Primary Modules
These modules are associated with the primary function of the Deep HyperNEAT (DHN) algorihtm.
### genome.py
Contains all functionality of the genome, a Compositional Pattern Producing Network (CPPN), its mutation operators, and information.
### phenomes.py
Contains multiple representations for feed-forward and recurrent neural networks for the CPPN and the Substrate.
### population.py
Contains all functionality and information of the populations used in DHN.
### activations.py
A library of activation functions used for the CPPN and Substrate phenomes.
### reproduction.py
Contains all functionality needed for the reproductive behavior in DHN.
### species.py
Contains functionality necessary for speciation in DHN.
### stagnation.py
Contains functionality necessary for stagnation schemes used in speciation.
### decode.py
Contains functionality to decode a given CPPN into a Substrate.

## Secondary Modules

### visualize.py
Allows you to visualize a CPPN or Substrate as a graph.
### math_util.py
Various, common math functions used throughout DHN.
### six_util.py
Various, common functions used to operate on lists and dicts throughout DHN.
