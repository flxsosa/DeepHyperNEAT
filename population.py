'''
Population class for Deep HyperNEAT

Felix Sosa
'''
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt 

from genome import Genome
from reproduction import Reproduction
from six_util import iteritems,itervalues,iterkeys

class Population():

	def __init__(self, key, size, elitism=1, state=None):
		self.key = key
		self.size = size
		self.best_genome = None
		self.last_best = 0
		self.elitism = elitism
		self.reproduction = Reproduction()
		if state == None:
			# Create new population
			self.population = self.reproduction.create_new_population(self.size)
		else:
			# Assign values from state
			self.population, self.reproduction = state
			
	def run_without_speciation(self, fitness_function, generations=None):
		current_gen = 0
		goal = 0.97
		reached_goal = False
		# Plot data
		best_y = []
		# Iterate for a number of generations or infinitely
		while current_gen < generations and not reached_goal:
			# Average fitness
			avg_fitness = 0
			# Assess fitness of current population
			fitness_function(list(iteritems(self.population)))
			# Find best genome in current generation and update avg fitness
			curr_best = None
			for genome in itervalues(self.population):
				avg_fitness += genome.fitness
				if curr_best is None or genome.fitness > curr_best.fitness:
					curr_best = genome
			# Update global best genome if possible
			if self.best_genome is None or curr_best.fitness > self.best_genome.fitness:
				self.best_genome = curr_best
			best_y.append(self.best_genome.fitness)
			# Reached fitness goal, we can stop
			if self.best_genome.fitness > goal:
				reached_goal = True

			# Return new population and continue to next generation
			print("\nCurrent Generation: {}".format(current_gen))
			print("Current Best Fitness: {}".format(self.best_genome.fitness))
			print("From Genome {}".format(self.best_genome.key))
			print("Average Fitness: {}".format(avg_fitness/self.size))
			for node in self.best_genome.nodes.values():
				print("Node {0} of type {1}".format(node.key, node.activation))
			for conn in self.best_genome.connections.values():
				print("Conn {0} of weight {1}".format(conn.key, conn.weight))

			# Order population by descending fitness
			pop_by_fitness = list(iteritems(self.population))
			pop_by_fitness.sort(reverse=True,key=lambda x: x[1].fitness)
			# Create new population
			new_population = {}
			new_pop_to_add = {}
			# Preserve elites of current population
			for gid, genome in pop_by_fitness[:self.elitism]:
				new_population[gid] = genome
			new_population_keys = list(iterkeys(new_population))
			print(new_population_keys)
			# Fill in rest of new population with mutants
			for _ in range(self.size - self.elitism):
				# Randomly choose elite to mutate
				idx = np.random.choice(new_population.keys())
				mutant_key = next(self.reproduction.genome_indexer)
				mutant = Genome(mutant_key)
				mutant.copy(new_population[idx])
				# print("Same?: {}".format(new_population[idx] is mutant))
				mutant.mutate()
				new_pop_to_add[mutant_key] = mutant
			new_population.update(new_pop_to_add)
			self.population = new_population
			current_gen += 1

		return self.best_genome
		
	def run_without_speciation_with_stag(self, fitness_function, generations=None):
		current_gen = 0
		goal = 0.97
		reached_goal = False
		stagnation = 20
		# Plot data
		best_y = []
		# Iterate for a number of generations or infinitely
		while current_gen < generations and not reached_goal:
			# Average fitness
			avg_fitness = 0
			# Assess fitness of current population
			fitness_function(list(iteritems(self.population)))
			# Find best genome in current generation and update avg fitness
			curr_best = None
			for genome in itervalues(self.population):
				avg_fitness += genome.fitness
				if curr_best is None or genome.fitness > curr_best.fitness:
					curr_best = genome
			# Update global best genome if possible
			if self.best_genome is None or curr_best.fitness > self.best_genome.fitness:
				self.best_genome = curr_best
			best_y.append(self.best_genome.fitness)
			# Reached fitness goal, we can stop
			if self.best_genome.fitness > goal:
				reached_goal = True
			# Order population by descending fitness
			pop_by_fitness = list(iteritems(self.population))
			pop_by_fitness.sort(reverse=True,key=lambda x: x[1].fitness)
			# Create new population
			new_population = {}
			
			if self.last_best > stagnation:
				self.population = self.reproduction.create_new_population(self.size)
				print("\nStagnated, replacing population")
				self.last_best = 0
				self.best_genome = None
				# Return new population and continue to next generation
				print("Current Generation: {}".format(current_gen))
				print("Average Fitness: {}".format(avg_fitness/self.size))
			else:
				# Preserve elites of current population
				for gid, genome in pop_by_fitness[:self.elitism]:
					new_population[gid] = genome
				new_population_keys = list(iterkeys(new_population))
				# Fill in rest of new population with mutants
				for _ in range(self.size - self.elitism):
					# Randomly choose elite to mutate
					idx = np.random.choice(new_population.keys())
					mutant_key = next(self.reproduction.genome_indexer)
					mutant = Genome(mutant_key)
					mutant.copy(new_population[idx])
					mutant.mutate()
					new_population[mutant_key] = mutant
				self.population = new_population
				# Return new population and continue to next generation
				print("\nCurrent Generation: {}".format(current_gen))
				print("Current Best Fitness: {}".format(self.best_genome.fitness))
				print("From Genome {}".format(self.best_genome.key))
				print("Average Fitness: {}".format(avg_fitness/self.size))
				for node in self.best_genome.nodes.values():
					print("Node {0} of type {1}".format(node.key, node.activation))
				for conn in self.best_genome.connections.values():
					print("Conn {0} of weight {1}".format(conn.key, conn.weight))
			
			current_gen += 1
			self.last_best += 1
		
		if reached_goal:
			best_x = range(current_gen)
			plt.plot(best_x, best_y)
			plt.yticks([y/10.0 for y in range(11)])
			plt.ylabel("Fitness")
			plt.xlabel("Generation")
			plt.show()
		return self.best_genome
