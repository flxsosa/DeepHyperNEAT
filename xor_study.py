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
			inputs = inputs + (0.0,)
			actual_output = substrate.activate(inputs)[0]
			sum_square_error += ((actual_output - expected)**2.0)/4.0
		genome.fitness = 1.0 - sum_square_error

# Gather winning and max complexity genome
winner_genome = pop.run_with_speciation(xor,num_generations)
max_genome = pop.max_complex_genome
output_file = open("reports/generation_output.txt", "w")
for n,g in winner_genome.prev_genomes:
	c = CPPN.create(g)
	s = decode(c, sub_in_dims, sub_o_dims, sub_sh_dims)
	sum_square_error = 0.0
	for inputs, expected in zip(xor_inputs, expected_outputs):
		print("Genome From Generation {}".format(n))
		output_file.write("Genome From Generation {}\n".format(n))
		print("Input: {}\nExpected Output: {}".format(inputs,expected))
		output_file.write("Input: {}\nExpected Output: {}\n".format(inputs,expected))
		inputs = inputs + (0.0,)
		actual_output = s.activate(inputs)[0]
		sum_square_error += ((actual_output - expected)**2.0)/4.0
		print("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
		output_file.write("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
	print("Total Loss: {}".format(sum_square_error))
	output_file.write("Total Loss: {}\n\n".format(sum_square_error))
	draw_net(c, filename="reports/champion_images/{}_cppn".format(n))
	draw_net(s, filename="reports/champion_images/{}_substrate".format(n))

# Decode winner into CPPN and Substrate
cppn = CPPN.create(winner_genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
report_ancestry(winner_genome)
draw_net(cppn, filename="reports/champion_images/xor_cppn")
draw_net(substrate, filename="reports/champion_images/xor_substrate")

# Decode max complexity genome into CPPN and Substrate
max_cppn = CPPN.create(max_genome)
max_substrate = decode(max_cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
report_ancestry(max_genome, "complex")
draw_net(max_cppn, filename="reports/complex_images/xor_max_cppn")
draw_net(max_substrate, filename="reports/complex_images/xor_max_substrate")

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