'''
Set of functions for reporting status of an evolutionary
run.

Felix Sosa
'''
from six_util import iteritems,itervalues,iterkeys

def report_fitness(pop):
	# Report average, min, and max fitness of a population
	avg_fitness = 0
	# Find best genome in current generation and update avg fitness
	for genome in itervalues(pop.population):
		avg_fitness += genome.fitness
	print("\n=========================================")
	print("\t     Generation: {}".format(pop.current_gen))
	print("=========================================")
	print("Best Fitness \t Avg Fitness \t Champion")
	print("============ \t =========== \t ========")
	print("{:.2f} \t\t {:.2f} \t\t {}".format(pop.best_genome.fitness, avg_fitness/pop.size,pop.best_genome.key))

def report_species(species_set):
	print("\nSpecies Key \t Sp. Fitness \t Sp. Size")
	print("=========== \t =========== \t ========")
	for species in species_set.species:
		print("{} \t\t {:.2} \t\t {}".format(species, 
			species_set.species[species].fitness,
			len(species_set.species[species].members)))

def report_ancestry(genome):
	print("\n\t Ancestry")
	print("===========================================")
	for mutation in genome.ancestry:
		print(mutation)