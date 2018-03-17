'''
Class for maintaining and implementing reproductive behavior in 
Deep HyperNEAT

Felix Sosa
'''
from genome import Genome
from itertools import count

class Reproduction():

	def __init__(self):
		self.genome_indexer = count(1)

	def create_new_population(self, num_genomes):
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
		"""Compute the proper number of offspring per species (proportional to fitness)."""
		
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
				curr_size += c
			# If species marginally improves, increase size by 1
			elif difference > 0:
				curr_size += 1
			elif difference < 0:
				curr_size -= 1

			species_sizes.append(curr_size)

		# Normalize the amounts so that the next generation is roughly
		# the population size requested by the user.
		total_spawn = sum(species_sizes)
		norm = pop_size / total_spawn
		species_sizes = [max(min_species_size, int(round(n * norm))) for n in species_sizes]

		return species_sizes

	def reproduce(self, config, species, pop_size, generation):
		"""
		Handles creation of genomes, either from scratch or by sexual or
		asexual reproduction from parents.
		"""
		# TODO: I don't like this modification of the species and stagnation objects,
		# because it requires internal knowledge of the objects.

		# Filter out stagnated species, collect the set of non-stagnated
		# species members, and compute their average adjusted fitness.
		# The average adjusted fitness scheme (normalized to the interval
		# [0, 1]) allows the use of negative fitness values without
		# interfering with the shared fitness scheme.
		all_fitnesses = []
		remaining_species = []
		for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
			if stagnant:
				self.reporters.species_stagnant(stag_sid, stag_s)
			else:
				all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))
				remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {} # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses) # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size,self.reproduction_config.elitism)
        spawn_amounts = self.compute_species_sizes(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population