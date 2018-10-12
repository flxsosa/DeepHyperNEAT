from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
from visualize import draw_net

# Substrate parameters
sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1

# Task parameters
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]

# Evolutionary parameters
pop_key = 0
pop_size = 150
pop_elitism = 2
num_generations = 500

# Define task
def xor(genomes):
	for genome_key, genome in genomes:
		cppn = CPPN.create(genome)
		substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
		sum_square_error = 0.0
		for inputs, expected in zip(xor_inputs, expected_outputs):
			inputs = inputs + (1.0,)
			actual_output = substrate.activate(inputs)[0]
			sum_square_error += ((actual_output - expected)**2.0)/4.0
		genome.fitness = 1.0 - sum_square_error

# Inititalize population
pop = Population(pop_key, pop_size,pop_elitism)

# Run population on the defined task for the specified number of generations
#	and collect the winner
winner_genome = pop.run_with_speciation(xor,num_generations)

# Decode winner genome into CPPN representation
cppn = CPPN.create(winner_genome)

# Decode Substrate from CPPN
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)

# Visualize networks of CPPN and Substrate. Files are saved in 
# 	reports/champion_images
draw_net(cppn, filename="reports/champion_images/xor_cppn")
draw_net(substrate, filename="reports/champion_images/xor_substrate")

# Run winning genome on the task again
print("\nChampion Genome: {} with Fitness {}\n".format(winner_genome.key, 
	  											winner_genome.fitness))
xor([(winner_genome.key,winner_genome)])