# Deep HyperNEAT: Open-ended evolution of deep neural networks
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Using DeepHyperNEAT
In order to run DHN in its current form, you need to create a task file. For reference, see xor_study.py.

This task file must contain:
- Necessary imports:
	```python
	from genome import Genome # Genome class
	from population import Population # Population class
	from phenomes import FeedForwardCPPN # CPPN class
	from decode import decode # Decoder for CPPN -> Substrate
	from visualize import draw_net # optional, for visualizing networks
	```
- Substrate parameters
	* Input dimensions
	* Output dimensions
	* Sheet dimensions (optional)
	```python
	sub_in_dims = [1,2] # Is of type list
	sub_sh_dims = [1,3] # Is of type list
	sub_o_dims = 1 # Is of type integer
	```
- Evolutionary parameters
	* Population size
	* Population elitism
	* Max number of generations
	```python
	pop_key = 0 # Key for population
	pop_size = 150
	pop_elitism = 2 # Number of members of pop to keep each generation
	```
- The task (defined as a function in python)
	* Task parameters:
		* Task inputs
		* Expected outputs (optional)
	```python
	def task(genomes):
		task_inputs = [1,2,3]
		expected_outputs = [2,4,6]
		for key, genome in genomes:
			cppn = CPPN.create(genome) # Create cppn from genome
			substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims) # Decode cppn into substrate
			error = 0.0 # Initialize error for current genome
			for inputs, expected in zip(xor_inputs, expected_outputs):
				inputs = inputs + (1.0,) # Append inputs with bias value
				actual_output = substrate.activate(inputs)[0] # Query substrate
				error += error_func(actual_output,expected) # Evaluate error
			genome.fitness = 1.0 - error # Assign fitness
	```
- A call to DHN to attempt to solve the task
	```python
	pop = Population(pop_key, pop_size, pop_elitism)
	solution = pop.run(task,num_generations)
	```

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