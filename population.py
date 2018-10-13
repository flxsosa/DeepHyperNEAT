'''
Population class for Deep HyperNEAT

Felix Sosa
'''
import numpy as np
from genome import Genome
from reproduction import Reproduction
from util import iteritems,itervalues
from species import SpeciesSet
from reporters import report_fitness, report_species, plot_fitness, plot_complexity, report_output

class Population():

	def __init__(self, key, size, elitism=1, state=None):
		self.key = key
		self.size = size
		self.best_genome = None
		self.max_complex_genome = None
		self.min_complex_genome = None
		self.avg_complexity = None
		self.max_dict = {}
		self.last_best = 0
		self.current_gen = 0
		self.elitism = elitism
		self.reproduction = Reproduction()
		self.species = SpeciesSet(3.5)

		if state == None:
			# Create new population
			self.population = self.reproduction.create_new_population(self.size)
			self.species.speciate(self.population,0)
		else:
			# Assign values from state
			self.population, self.reproduction = state	

	def run(self,task,goal,generations=None):
		'''
		Run evolution on a given task for a number of generations or until
		a goal is reached.

		task -- the task to be solved
		goal -- the goal to reach for the given task that defines a solution
		generations -- the max number of generations to run evolution for
		'''
		self.current_gen = 0
		reached_goal = False
		# Plot data
		best_fitnesses = []
		max_complexity = []
		min_complexity = []
		avg_complexity = []
		while self.current_gen < generations and not reached_goal:
			# Assess fitness of current population
			task(list(iteritems(self.population)))
			# Find best genome in current generation and update avg fitness
			curr_best = None
			curr_max_complex = None
			curr_min_complex = None
			avg_complexities = 0
			for genome in itervalues(self.population):
				avg_complexities += genome.complexity()
				# Update generation's most fit
				if curr_best is None or genome.fitness > curr_best.fitness:
					curr_best = genome
				# Update generation's most complex
				if curr_max_complex is None or genome.complexity() > curr_max_complex.complexity():
					curr_max_complex = genome	
				# Update generation's least complex
				if curr_min_complex is None or genome.complexity() < curr_min_complex.complexity():
					curr_min_complex = genome

			# Update global best genome if possible
			if self.best_genome is None or curr_best.fitness > self.best_genome.fitness:
				self.best_genome = curr_best
			
			# Update global most and least complex genomes
			if self.max_complex_genome is None or curr_max_complex.complexity() > self.max_complex_genome.complexity():
				self.max_complex_genome = curr_max_complex
			if self.min_complex_genome is None or curr_min_complex.complexity() < self.min_complex_genome.complexity():
				self.min_complex_genome = curr_min_complex

			self.max_dict[self.current_gen] = self.max_complex_genome

			# Reporters
			report_fitness(self)
			report_species(self.species, self.current_gen)
			report_output(self)
			best_fitnesses.append(self.best_genome.fitness)
			max_complexity.append(self.max_complex_genome.complexity())
			min_complexity.append(self.min_complex_genome.complexity())
			avg_complexity.append((avg_complexities+0.0)/len(self.population))
			self.avg_complex = (avg_complexities+0.0)/len(self.population)
			avg_complexities = 0

			# Reached fitness goal, we can stop
			if self.best_genome.fitness >= goal:
				reached_goal = True
			
			# Create new unspeciated popuation based on current population's fitness
			self.population = self.reproduction.reproduce_with_species(self.species,
																	   self.size, 
																	   self.current_gen)
			# Check for species extinction (species did not perform well)
			if not self.species.species:
				print("!!! Species went extinct !!!")
				self.population = self.reproduction.create_new_population(self.size)
		
			# Speciate new population
			self.species.speciate(self.population, self.current_gen)
			self.current_gen += 1
		
		generations = range(self.current_gen)
		plot_fitness(generations, best_fitnesses)
		return self.best_genome