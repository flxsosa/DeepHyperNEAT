'''
Class for maintaining and implementing reproductive behavior in Deep HyperNEAT.

Largely copied from neat-python. (Copyright 2015-2017, CodeReclaimers, LLC.)
'''
import random
from math import ceil
from genome import Genome
from stagnation import Stagnation
from itertools import count
from util import itervalues, iteritems, mean

class Reproduction:

	def __init__(self):
		self.genome_indexer = count(1)
		self.reporters = None
		# Number of elites allowed to be cloned into species each gen
		self.species_elitism = 1
		self.stagnation = Stagnation(self.species_elitism)
		# Fraction of members of a species allowed to reproduce each gen
		self.species_reproduction_threshold = 0.2

	def create_new_population(self, num_genomes):
		'''
		Creates a fresh population

		num_genomes -- number of genomes to create for the population
		'''
		new_genomes = {}
		# Create n new, minimal genomes
		for i in range(num_genomes):
			gid = next(self.genome_indexer)
			# Create genome
			new_genome = Genome(gid)
			new_genomes[gid] = new_genome

		return new_genomes

	@staticmethod
	def compute_species_sizes(adjusted_fitness, previous_sizes, pop_size, min_species_size):
		'''
		Compute the proper number of offspring per species (proportional to fitness).

		adjusted_fitness -- normalized fitness of members in the population
		previous_sizes   -- previous sizes of the species
		pop_size 		 -- population size
		min_species_size -- minimum species size
		'''
		
		adujst_fitness_sum = sum(adjusted_fitness)
		species_sizes = []
		for adjusted_fit, prev_size in zip(adjusted_fitness, previous_sizes):
			if adujst_fitness_sum > 0:
				# Species size should be proportional to fitness if positive
				species_size = max(min_species_size, adjusted_fit/adujst_fitness_sum*pop_size)
			else:
				species_size = min_species_size
			# This is basically determining if the species improved in fitness or 
			# decreased
			difference = (species_size-prev_size)*0.5
			count = int(round(difference))
			curr_size = prev_size
			# If species sees large increase in fitness, increase accordingly
			if abs(count) > 0:
				curr_size += count
			# If species marginally improves, increase size by 1
			elif difference > 0:
				curr_size += 1
			elif difference < 0:
				curr_size -= 1

			species_sizes.append(curr_size)

		# Normalize the amounts so that the next generation is roughly
		# the population size requested by the user.
		total_spawn = sum(species_sizes)
		norm = pop_size/total_spawn
		species_sizes = [max(min_species_size, int(round(n * norm))) for n in species_sizes]
		# print(species_sizes)
		return species_sizes

	def reproduce_with_species(self, species_set, pop_size, generation):
		'''
		Creates and speciates genomes.

		species_set -- set of current species
		pop_size    -- population size
		generation  -- current generation
		'''
		all_fitnesses = []
		remaining_species = []
		# Traverse species and grab fitnesses from non-stagnated species
		for sid, species, species_is_stagnant in self.stagnation.update(species_set, 
																		generation):
			if species_is_stagnant:
				print("!!! Species {} Stagnated !!!".format(sid))
				# self.reporters.species_stagnant(sid, species)
				pass
			else:
				# Add fitnesses of members of current species
				all_fitnesses.extend(member.fitness for member in 
									 itervalues(species.members))
				remaining_species.append(species)
		# No species
		if not remaining_species:
			species_set.species = {}
			return {}
		# Find min/max fitness across entire population
		min_population_fitness = min(all_fitnesses)
		max_population_fitness = max(all_fitnesses)
		# Do not allow the fitness range to be zero, as we divide by it below.
		population_fitness_range = max(1.0, max_population_fitness - min_population_fitness)
		# Compute adjusted fitness and record minimum species size
		for species in remaining_species:
			# Determine current species average fitness
			mean_species_fitness = mean([member.fitness for member in 
										itervalues(species.members)])
			max_species_fitness = max([member.fitness for member in 
										itervalues(species.members)])
			# Determine current species adjusted fitness and update it
			species_adjusted_fitness = (mean_species_fitness-
				  min_population_fitness)/population_fitness_range
			species.adjusted_fitness = species_adjusted_fitness
			species.max_fitness = max_species_fitness
		adjusted_fitnesses = [species.adjusted_fitness for species in remaining_species]
		avg_adjusted_fitness = mean(adjusted_fitnesses)
		# Compute the number of new members for each species in the new generation.
		previous_sizes = [len(species.members) for species in remaining_species]
		min_species_size = max(2, self.species_elitism)
		spawn_amounts = self.compute_species_sizes(adjusted_fitnesses, previous_sizes,
												   pop_size, min_species_size)
		new_population = {}
		species_set.species = {}
		for spawn, species in zip(spawn_amounts, remaining_species):
			# If elitism is enabled, each species always at least gets to retain its elites.
			spawn = max(spawn, self.species_elitism)
			assert spawn > 0
			# The species has at least one member for the next generation, so retain it.
			old_species_members = list(iteritems(species.members))
			# Update species with blank slate
			species.members = {}
			# Update species in species set accordingly
			species_set.species[species.key] = species
			# Sort old species members in order of descending fitness.
			old_species_members.sort(reverse=True, key=lambda x: x[1].fitness)
			# Clone elites to new generation.
			if self.species_elitism > 0:
				for member_key, member in old_species_members[:self.species_elitism]:
					new_population[member_key] = member
					spawn -= 1
			# If the species only has room for the elites, move onto next species
			if spawn <= 0: continue
			# Only allow fraction of species members to reproduce
			reproduction_cutoff = int(ceil(self.species_reproduction_threshold *
									  len(old_species_members)))
			# Use at least two parents no matter what the threshold fraction result is.
			reproduction_cutoff = max(reproduction_cutoff, 2)
			old_species_members = old_species_members[:reproduction_cutoff]

			# Randomly choose parents and produce the number of offspring allotted to the species.
			# NOTE: Asexual reproduction for now
			while spawn > 0:
				spawn -= 1
				parent1_key, parent1 = random.choice(old_species_members)
				# parent2_key, parent2 = random.choice(old_species_members)
				child_key = next(self.genome_indexer)
				child = Genome(child_key)
				# child.crossover(parent1, parent2)
				child.copy(parent1,generation)
				child.mutate(generation)
				new_population[child_key] = child
		return new_population