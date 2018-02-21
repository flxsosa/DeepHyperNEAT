'''
Class for maintaining and implementing reproductive behavior in 
Deep HyperNEAT

Felix Sosa
'''
from genome import Genome
from itertools import count

def Reproduction():

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