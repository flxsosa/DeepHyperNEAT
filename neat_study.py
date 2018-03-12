from genome import Genome
from population import Population
from phenomes import FeedForwardNet as neural_net 
from decode import decode
from visualize import draw_net

# XOR I/O
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]

def xor(genomes):
	# Iterate through and evaluate candidate genomes
	for genome_key, genome in genomes:
		net = neural_net.create(genome)
		sum_square_error = 0.0
		for inputs, expected in zip(xor_inputs, expected_outputs):
			inputs = inputs
			actual_output = net.activate(inputs)[0]
			sum_square_error += ((actual_output - expected)**2.0)/4.0
		# Assign fitness to genome
		genome.fitness = 1.0 - sum_square_error

# Create a population
pop_key = 0
pop_size = 150
pop_elitism = 5
pop = Population(0,pop_size,pop_elitism)

# Run evolution and collect winner genome
num_generations = 100
winner_genome = pop.run_without_speciation(xor,num_generations)

# Decode winner into CPPN and Substrate
net = neural_net.create(winner_genome)
# Inform user of winning genome
print("\nWinner Genome: {0} with Fitness {1}".format(winner_genome.key, 
	  winner_genome.fitness))
# Init loss (SSE)
sum_square_error = 0.0
for inputs, expected in zip(xor_inputs, expected_outputs):
	print("Input: {}".format(inputs))
	print("Expected Output: {}".format(expected))
	inputs = inputs
	actual_output = net.activate(inputs)[0]
	print("Actual Output: {}".format(actual_output))
	sum_square_error += ((actual_output - expected)**2.0)/4.0
	print("Loss: {}".format(sum_square_error))
print("Total Loss: {}".format(sum_square_error))

draw_net(net, filename="images/ann")