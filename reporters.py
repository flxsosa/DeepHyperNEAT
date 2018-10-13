'''
Set of functions for reporting status of an evolutionary run.

NOTE: Only meant for XOR at the moment. Working on generalizing the framework
	  to any task.
'''
from util import iteritems,itervalues,iterkeys
from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
import seaborn
import matplotlib.pyplot as plt

sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]

def report_output(pop):
	'''
	Reports the output of the current champion for the xor task.

	pop -- population to be reported
	'''
	genome = pop.best_genome
	cppn = CPPN.create(genome)
	substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
	sum_square_error = 0.0
	print("\n=================================================")
	print("\tChampion Output at Generation: {}".format(pop.current_gen))
	print("=================================================")
	for inputs, expected in zip(xor_inputs, expected_outputs):
		print("Input: {}\nExpected Output: {}".format(inputs,expected))
		inputs = inputs + (1.0,)
		actual_output = substrate.activate(inputs)[0]
		sum_square_error += ((actual_output - expected)**2.0)/4.0
		print("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
	print("Total Loss: {}".format(sum_square_error))

def report_fitness(pop):
	'''
	Report average, min, and max fitness of a population

	pop -- population to be reported
	'''
	avg_fitness = 0
	# Find best genome in current generation and update avg fitness
	for genome in itervalues(pop.population):
		avg_fitness += genome.fitness
	print("\n=================================================")
	print("\t\tGeneration: {}".format(pop.current_gen))
	print("=================================================")
	print("Best Fitness \t Avg Fitness \t Champion")
	print("============ \t =========== \t ========")
	print("{:.2f} \t\t {:.2f} \t\t {}".format(pop.best_genome.fitness, 
		  avg_fitness/pop.size,pop.best_genome.key))
	print("=================================================")
	print("Max Complexity \t Avg Complexity")
	print("============ \t =========== \t ========")
	print("{} \t\t {}".format(None, pop.avg_complexity))

def report_species(species_set, generation):
	'''
	Reports species statistics

	species_set -- set contained the species
	generation  -- current generation
	'''
	print("\nSpecies Key \t Fitness Mean/Max \t Sp. Size")
	print("=========== \t ================ \t ========")
	for species in species_set.species:
		print("{} \t\t {:.2} / {:.2} \t\t {}".format(species, 
			species_set.species[species].fitness,
			species_set.species[species].max_fitness,
			len(species_set.species[species].members)))

def plot_fitness(x,y):
	plt.plot(x,y)
	plt.ylabel("Fitness")
	plt.xlabel("Generation")
	plt.tight_layout()
	plt.savefig("reports/fitness_plot.png")

def plot_complexity(x1,y1,y2,y3):
	plt.plot(x1,y1,x1,y2,x1,y3)
	plt.ylabel("Complexity")
	plt.xlabel("Generation")
	plt.legend(['?','max_comp', 'min_comp', 'avg_comp'])
	plt.tight_layout()
	plt.savefig("reports/complexity_plot.png")