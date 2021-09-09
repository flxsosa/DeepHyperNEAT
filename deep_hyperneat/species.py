'''
Classes that handle speciation in Deep HyperNEAT.

Largely copied from neat-python. Copyright 2015-2017, CodeReclaimers, LLC.
'''
from itertools import count
from deep_hyperneat.util import iteritems, iterkeys, itervalues

class GenomeDistanceCache:
	# Cache of genome distances
	def __init__(self):
		self.distances = {}
		self.hits = 0
		self.misses = 0
		self.compatibility_disjoint_coefficient = 1.0
		self.compatibility_weight_coefficient = 0.5

	def __call__(self, genome0, genome1):
		genome_key_0 = genome0.key
		genome_key_1 = genome1.key
		distance = self.distances.get((genome_key_0, genome_key_1))
		if distance is None:
			# Distance is not already computed.
			distance = self.genome_distance(genome0,genome1)
			self.distances[genome_key_0, genome_key_1] = distance
			self.distances[genome_key_1, genome_key_0] = distance
			self.misses += 1
		else:
			self.hits += 1
		return distance

	def genome_distance(self, genome0, genome1):
		'''
		Computes genome distance between two genomes
		'''
		node_distance = 0.0
		# Determine node distance
		if genome0.nodes or genome1.nodes:
			# Number of disjoing nodes between genomes
			disjoint_nodes = 0
			# Count number of disjoint node genes between genomes
			for genome_1_node_key in iterkeys(genome1.nodes):
				if genome_1_node_key not in genome0.nodes:
					disjoint_nodes += 1
			# Determine genetic distance between  individual node genes
			for genome_0_node_key, genome_0_node in iteritems(genome0.nodes):
				genome_1_node = genome1.nodes.get(genome_0_node_key)
				if genome_1_node is None:
					disjoint_nodes += 1
				else:
					# Homologous genes compute their own distance value.
					node_distance += self.node_gene_distance(genome_0_node,
														genome_1_node)
			# Find most number of nodes in either genome
			max_nodes = max(len(genome0.nodes), len(genome1.nodes))
			# Determine final node genetic distance
			node_distance = (node_distance +
								(self.compatibility_disjoint_coefficient *
								disjoint_nodes))/max_nodes

		# Determine connection gene distance
		connection_distance = 0.0
		if genome0.connections or genome1.connections:
			disjoint_connections = 0
			for genome_1_conn_key in iterkeys(genome1.connections):
				if genome_1_conn_key not in genome0.connections:
					disjoint_connections += 1

			for genome_0_conn_key, genome_0_conn in iteritems(genome0.connections):
				genome_1_conn = genome1.connections.get(genome_0_conn_key)
				if genome_1_conn is None:
					disjoint_connections += 1
				else:
					# Homologous genes compute their own distance value.
					connection_distance += self.connection_gene_distance(genome_0_conn,
																	genome_1_conn)

			max_conn = max(len(genome0.connections), len(genome1.connections))
			connection_distance = (connection_distance +
									(self.compatibility_disjoint_coefficient *
									disjoint_connections)) / max_conn

		distance = node_distance + connection_distance
		return distance

	def node_gene_distance(self, node_gene_0, node_gene_1):
		'''
		Computes genetic distance between node genes
		'''
		distance = abs(node_gene_0.bias-node_gene_1.bias)
		if node_gene_0.activation != node_gene_1.activation:
			distance += 1.0
		return distance * self.compatibility_weight_coefficient

	def connection_gene_distance(self, conn_gene_0, conn_gene_1):
		'''
		Computes genetic distance between connection genes
		'''
		d = abs(conn_gene_0.weight - conn_gene_1.weight)
		return d * self.compatibility_weight_coefficient

class Species:
	# Class for individual species
	def __init__(self, key, generation):
		self.key = key
		self.created = generation
		self.last_improved = generation
		self.representative = None
		self.members = {}
		self.fitness = None
		self.max_fitness = None
		self.adjusted_fitness = None
		self.fitness_history = []

	def update(self, representative, members):
		self.representative = representative
		self.members = members

	def get_fitnesses(self):
		return [m.fitness for m in itervalues(self.members)]

class SpeciesSet:
	# Class for handling sets of species within a population
	def __init__(self, threshold):
		self.threshold = threshold
		self.species = {}
		self.species_indexer = count(1)
		self.genome_to_species = {}

	def speciate(self,population,generation):
		'''
		Speciates a population.
		'''
		# Compatibility threshold
		compatibility_threshold = self.threshold
		# Set of unspeciated members of the population
		unspeciated = set(iterkeys(population))
		# Means of determining distances
		distances = GenomeDistanceCache()
		# New representatives and members of species
		new_representatives = {}
		new_members = {}
		# Traverse through set of species from last generation
		for sid, species in iteritems(self.species):
			# Candidates for current species representatives
			candidate_representatives = []
			# Traverese genomes in the unspeciated and check their distance
			# from the current species representative
			for gid in unspeciated:
				genome = population[gid]
				genome_distance = distances(species.representative, genome)
				candidate_representatives.append((genome_distance, genome))
			# The new representative for the current species is the
			# closest to the current representative
			_, new_rep = min(candidate_representatives, key=lambda x: x[0])
			new_rid = new_rep.key
			new_representatives[sid] = new_rid
			new_members[sid] = [new_rid]
			unspeciated.remove(new_rid)

		# Partition the population in species based on genetic similarity
		while unspeciated:
			gid = unspeciated.pop()
			genome = population[gid]
			# Find the species with the most similar representative to the
			# 	current genome from the unspeciated set
			candidate_species = []
			# Traverse species and their representatives
			for sid, rid in iteritems(new_representatives):
				representative = population[rid]
				# Determine current genome's distance from representative
				genome_distance = distances(representative, genome)
				# If it's below threshold, add it to list for adding to the species
				if genome_distance < compatibility_threshold:
					candidate_species.append((genome_distance, sid))
			# Add current genome to the species its most genetically similar to
			if candidate_species:
				_, sid = min(candidate_species, key=lambda x: x[0])
				new_members[sid].append(gid)
			else:
				# No species is similar enough so we create a mnew species with
				# 	the current genome as its representative
				sid = next(self.species_indexer)
				new_representatives[sid] = gid
				new_members[sid] = [gid]
			# Update species collection based on new speciation
			self.genome_to_species = {}
			for sid, rid in iteritems(new_representatives):
				# Add species if not existing in current species set
				s = self.species.get(sid)
				if s is None:
					s = Species(sid, generation)
					self.species[sid] = s
				# Collect and add members to current species
				members = new_members[sid]
				for gid in members:
					self.genome_to_species[gid] = sid
				# Update current species members and represenative
				member_dict = {gid:population[gid] for gid in members}
				s.update(population[rid], member_dict)

	def get_species_key(self, key):
		return self.genome_to_species[key]

	def get_species(self, key):
		sid = self.genome_to_species[key]
		return self.species[sid]
