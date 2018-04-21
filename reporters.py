'''
Set of functions for reporting status of an evolutionary
run.

Felix Sosa
'''
from six_util import iteritems,itervalues,iterkeys
from visualize import draw_net
from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
import seaborn
import matplotlib.pyplot as plt

fitness_file = open("reports/terminal_output.txt", "w")
champion_ancestry_file = open("reports/champion_ancestry.txt", "w")
complex_ancestry_file = open("reports/complex_ancestry.txt", "w")

sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]

def report_output(pop):
	genome = pop.best_genome
	cppn = CPPN.create(genome)
	substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
	sum_square_error = 0.0
	print("\n=================================================")
	print("\tChampion Output at Generation: {}".format(pop.current_gen))
	print("=================================================")
	fitness_file.write("\n=================================================\n")
	fitness_file.write("\t Champion Output at Generation: {}\n".format(pop.current_gen))
	fitness_file.write("=================================================\n")
	for inputs, expected in zip(xor_inputs, expected_outputs):
		print("Input: {}\nExpected Output: {}".format(inputs,expected))
		fitness_file.write("Input: {}\nExpected Output: {}\n".format(inputs,expected))
		inputs = inputs + (1.0,)
		actual_output = substrate.activate(inputs)[0]
		sum_square_error += ((actual_output - expected)**2.0)/4.0
		print("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
		fitness_file.write("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
	print("Total Loss: {}".format(sum_square_error))
	fitness_file.write("Total Loss: {}\n".format(sum_square_error))

def report_fitness(pop):
	# Report average, min, and max fitness of a population
	avg_fitness = 0
	# Find best genome in current generation and update avg fitness
	for genome in itervalues(pop.population):
		avg_fitness += genome.fitness
	print("\n=================================================")
	print("\t\tGeneration: {}".format(pop.current_gen))
	print("=================================================")
	print("Best Fitness \t Avg Fitness \t Champion")
	print("============ \t =========== \t ========")
	print("{:.2f} \t\t {:.2f} \t\t {}".format(pop.best_genome.fitness, avg_fitness/pop.size,pop.best_genome.key))
	fitness_file.write("\n=================================================\n")
	fitness_file.write("\t\tGeneration: {}\n".format(pop.current_gen))
	fitness_file.write("=================================================\n")
	fitness_file.write("Best Fitness \t Avg Fitness \t Champion\n")
	fitness_file.write("============ \t =========== \t ========\n")
	fitness_file.write("{:.2f} \t\t {:.2f} \t\t {}\n".format(pop.best_genome.fitness, avg_fitness/pop.size,pop.best_genome.key))

def report_species(species_set, generation, graphics=True):
	print("\nSpecies Key \t Fitness Mean/Max \t Sp. Size")
	print("=========== \t ================ \t ========")
	fitness_file.write("\nSpecies Key \t Fitness Mean/Max \t Sp. Size\n")
	fitness_file.write("=========== \t ================ \t ========\n")
	for species in species_set.species:
		print("{} \t\t {:.2} / {:.2} \t\t {}".format(species, 
			species_set.species[species].fitness,
			species_set.species[species].max_fitness,
			len(species_set.species[species].members)))
		fitness_file.write("{} \t\t {:.2} / {:.2} \t\t {}\n".format(species, 
			species_set.species[species].fitness,
			species_set.species[species].max_fitness,
			len(species_set.species[species].members)))
		if graphics:
			cppn = CPPN.create(species_set.species[species].representative)
			draw_net(cppn, filename="reports/species_images/gen_{}_cppn_species_{}".format(generation, species))

def report_ancestry(genome, genome_type="champion"):
	file = champion_ancestry_file if genome_type == "champion" else complex_ancestry_file
	# print("\n\t Ancestry")
	# print("===========================================")
	file.write("\t Ancestry\n")
	file.write("===========================================\n")
	for mutation in genome.ancestors:
		# print(mutation)
		file.write("{}\n".format(mutation))

def plot_fitness(x,y):
	plt.plot(x,y)
	plt.ylabel("Fitness")
	plt.xlabel("Generation")
	# plt.legend(['Best Fit', 'Avg Fit'])
	plt.tight_layout()
	plt.savefig("reports/fitness_plot.png")

def plot_complexity(x1,y1,y2,y3):
	plt.plot(x1,y1,x1,y2,x1,y3)
	plt.ylabel("Complexity")
	plt.xlabel("Generation")
	plt.legend(['?','max_comp', 'min_comp', 'avg_comp'])
	plt.tight_layout()
	plt.savefig("reports/complexity_plot.png")