'''
Population class for Deep HyperNEAT

Felix Sosa
'''
import reproduction

from itertools import iteritems,itervalues

class Population():

	def __init__(self, id, size, elitism=1, state=None):
		self.id = id
		self.size = size
		self.best_genome = None
		self.elitism = elitism
		self.reproduction = reproduction.Reproduction()

		if state == None:
			# Create new population
			self.population = self.reproduction.create_new(self.size)
		else:
			# Assign values from state
			self.population, self.reproduction = state

	def run_without_speciation(self, fitness_function, generations=None):
		'''
		Run the Deep HyperNEAT algorithm without speciation as described in NEAT
		'''
		current_gen = 0

		# Iterate for a number of generations or infinitely
		while n is None or current_gen < generations:
			# Assess fitness of current population
			fitness_function(list(iteritems(self.population)))

			# Find best genome in current generation
			curr_best = None
			for genome in itervalues(self.population):
				if curr_best is None or genome.fitness > curr_best.fitness:
					curr_best = genome

			# Update global best genome if possible
			if self.best_genome is None or curr_best.fitness > self.best_genome.fitness:
				self.best_genome = best

			# Order population by descending fitness
			pop_by_fitness = list(iteritems(self.population))
			pop_by_fitness.sort(reverse=True,key=lambda x: x[1].fitness)

			# Create new population
			new_population = {}

			# Preserve elites of current population
			for gid, genome in pop_by_fitness[:self.elitism]:
				new_population[gid] = genome

			# Fill in rest of new population with mutants

			# Return new population and continue to next generation
			self.population = new_population
			current_gen += 1


