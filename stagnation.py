'''
Maintains stagnation behavior for speciation in Deep HyperNEAT

Felix Sosa
March 17, 2018
'''
import sys
from math_util import mean
from six_util import iteritems

class Stagnation:
    # Stagnation class
    def __init__(self, species_elitism=3):
        self.species_fitness_func = mean
        self.reporters = None
        self.species_elitism = 3
        self.max_stagnation = 10

    def update(self, species_set, generation):
        # Updates species fitness history, checks for stagnated species,
        # and returns a list with stagnant species to remove
        species_data = []
        for sid, species in iteritems(species_set.species):
            if species.fitness_history:
                prev_fitness = max(species.fitness_history)
            else:
                prev_fitness = -sys.float_info.max
            species.fitness = self.species_fitness_func(species.get_fitnesses())
            species.fitness_history.append(species.fitness)
            species.adjusted_fitness = None
            if prev_fitness is None or species.fitness > prev_fitness:
                species.last_improved = generation
            species_data.append((sid, species))
        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)
        result = []
        species_fitnesses = []
        num_non_stagnant_species = len(species_data)
        for idx, (sid, species) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit
            stagnant_time = generation - species.last_improved
            is_stagnant = False
            if num_non_stagnant_species > self.species_elitism:
                is_stagnant = stagnant_time >= self.max_stagnation
            if (len(species_data) - idx) <= self.species_elitism:
                is_stagnant = False
            if is_stagnant:
                num_non_stagnant_species -= 1
            result.append((sid, species, is_stagnant))
            species_fitnesses.append(species.fitness)
        return result
