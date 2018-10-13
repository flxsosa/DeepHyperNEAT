'''
File for the study used to test DeepHyperNEAT in "Deep HyperNEAT: 
Evolving the Size and Depth ofthe Substrate" by Felix Sosa and 
Kenneth Stanley.

Just run "python paper_study.py" in terminal to rerun if desired.
'''
from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
import numpy as np

# Substrate parameters
input_dim = [1,2]
hidden_dim = [1,3]
output_dim = 1

# Evolutionary parameters
fitness_goal = 0.98
num_generations = 500
population_key = 0
population_size = 150
population_elitism = 15

# Champion fitnesses
champ_fitness = []
# Population generations
pop_gens = []

# Define task
def xor(genomes):
	# Task parameters
	task_input = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
	task_output = [0.0, 1.0, 1.0, 0.0]
	for genome_key, genome in genomes:
		cppn = CPPN.create(genome)
		substrate = decode(cppn, input_dim, output_dim, hidden_dim)
		sum_square_error = 0.0
		# Test substrate on task
		for inputs, expected in zip(task_input, task_output):
			inputs += 0.0,
			substrate_output = substrate.activate(inputs)[0]
			sum_square_error += ((substrate_output - expected)**2.0)/4.0
		genome.fitness = 1.0 - sum_square_error

# Evolutionary run
for _ in range(100):
	# Run task and gather winning genome
	pop = Population(population_key, population_size, population_elitism)
	champion = pop.run(xor,fitness_goal,num_generations)
	champ_fitness.append((champion.num_depth, champion.num_breadth))
	pop_gens.append(pop.current_gen)
	# Print to user
	print("\nChampion Genome: {} with Fitness {}\n".format(champion.key, 
		  											champion.fitness))

num_depth, num_breadth = 0,0
for x in champ_fitness:
	if x[0] != 0:
		num_depth += 1
	if x[1] != 0:
		num_breadth += 1

print("Number of IncDepth Mutations: {} out of {}".format(num_depth,100))
print("Number of IncBreadth Mutations: {} out of {}".format(num_breadth,100))
print("Mean Number of Generations to Solution: {} with StdDev: {}".format(np.mean(pop_gens), 
	  np.std(pop_gens)))
print("Mean Number of IncDepth Mutations per Champion: {}".format(np.mean([x[0] for x in champ_fitness])))
print("StdDev: {}".format(np.std([x[0] for x in champ_fitness])))
print("Mean Number of IncBreadth Mutation per Champion: {}".format(np.mean([x[1] for x in champ_fitness])))
print("StdDev: {}".format(np.std([x[1] for x in champ_fitness])))