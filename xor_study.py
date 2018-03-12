from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
from visualize import draw_net

# Substrate Parameters
sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]
# Evolution and population parameters
pop_key = 0
pop_size = 150
pop_elitism = 2
pop = Population(0, pop_size,pop_elitism)
num_generations = 100

def xor(genomes):
	# Iterate through and evaluate candidate genomes
	for genome_key, genome in genomes:
		# Convert genome into usable CPPN
		cppn = CPPN.create(genome)
		# Decode CPPN into substrate
		substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
		# Assess fitness
		sum_square_error = 0.0
		for inputs, expected in zip(xor_inputs, expected_outputs):
			inputs = inputs #+ (1.0,)
			actual_output = substrate.activate(inputs)[0]
			sum_square_error += ((actual_output - expected)**2.0)/4.0
		# Assign fitness to genome
		genome.fitness = 1.0 - sum_square_error

winner_genome = pop.run_without_speciation(xor,num_generations)

# Decode winner into CPPN and Substrate
cppn = CPPN.create(winner_genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)

print("Winner Genome: {0} with Fitness {1}".format(winner_genome.key, 
	  winner_genome.fitness))
print("Output Nodes: {}".format(winner_genome.output_keys))
for node in winner_genome.nodes.values():
	print("Node {0} of type {1}".format(node.key, node.activation))

# Convert genome into usable CPPN
cppn = CPPN.create(winner_genome)
# Decode CPPN into substrate
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
# Init loss (SSE)
sum_square_error = 0.0
for inputs, expected in zip(xor_inputs, expected_outputs):
	print("Input: {}".format(inputs))
	print("Expected Output: {}".format(expected))
	inputs = inputs #+ (1.0,)
	actual_output = substrate.activate(inputs)[0]
	print("Actual Output: {}".format(actual_output))
	sum_square_error += ((actual_output - expected)**2.0)/4.0
	print("Loss: {}".format(sum_square_error))
print("Total Loss: {}".format(sum_square_error))

draw_net(cppn, filename="images/dhn_cppn")
draw_net(substrate, filename="images/dhn_substrate")