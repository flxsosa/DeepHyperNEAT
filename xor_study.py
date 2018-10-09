from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
from visualize import draw_net
from reporters import report_ancestry

sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]
pop_key = 0
pop_size = 150
pop_elitism = 2
pop = Population(0, pop_size,pop_elitism)
num_generations = 500

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

# Gather winning and max complexity genome
winner_genome = pop.run_with_speciation(xor,num_generations)

# Decode winner into CPPN and Substrate
cppn = CPPN.create(winner_genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
report_ancestry(winner_genome)
draw_net(cppn, filename="reports/champion_images/xor_cppn")
draw_net(substrate, filename="reports/champion_images/xor_substrate")

print("\nChampion Genome: {} with Fitness {}\n".format(winner_genome.key, 
	  											winner_genome.fitness))
sum_square_error = 0.0
for inputs, expected in zip(xor_inputs, expected_outputs):
	print("Input: {}\nExpected Output: {}".format(inputs,expected))
	inputs = inputs + (0.0,)
	actual_output = substrate.activate(inputs)[0]
	sum_square_error += ((actual_output - expected)**2.0)/4.0
	print("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
print("Total Loss: {}".format(sum_square_error))