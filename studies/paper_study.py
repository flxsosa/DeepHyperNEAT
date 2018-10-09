from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
import numpy as np

# Substrate parameters
input_dim = [1,2]
hidden_dim = [1,3]
output_dim = 1

# Task parameters
task_input = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
task_output = [0.0, 1.0, 1.0, 0.0]

# Evolutionary parameters
num_generations = 500
population_key = 0
population_size = 150
population_elitism = 15

# Define task
def xor(genomes):
	for genome_key, genome in genomes:
		# Decode genome into cppn
		cppn = CPPN.create(genome)
		# Decode cppn into substrate
		substrate = decode(cppn, input_dim, output_dim, hidden_dim)
		# Error 
		sum_square_error = 0.0
		# Test substrate on task
		for inputs, expected in zip(task_input, task_output):
			inputs += 0.0,
			substrate_output = substrate.activate(inputs)[0]
			sum_square_error += ((substrate_output - expected)**2.0)/4.0
		genome.fitness = 1.0 - sum_square_error

# Champion fitnesses
champ_fitness = []
# Population generations
pop_gens = []

# Define evolutionary run
for _ in range(100):
	# Run task and gather winning genome
	pop = Population(population_key, population_size, population_elitism)
	champion = pop.run_with_speciation(xor,num_generations)
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

print("Num depth is {} out of {}".format(num_depth,100))
print("Num breadth is {} out of {}".format(num_breadth,100))
print("Mean Fitness: {}".format(np.mean(pop_gens)))
print("Standard DEviation: {}\n".format(np.std(pop_gens)))
print("Mean Depth: {}".format(np.mean([x[0] for x in champ_fitness])))
print("Std Depth: {}".format(np.std([x[0] for x in champ_fitness])))
print("Mean Breadth: {}".format(np.mean([x[1] for x in champ_fitness])))
print("Std Breadth: {}".format(np.std([x[1] for x in champ_fitness])))