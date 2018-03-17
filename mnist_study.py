import csv
import matplotlib.pyplot as plt
import numpy as np
from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
from visualize import draw_net

# open the CSV file and read its contents into a list
data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# substrate parameters
sub_input = [28,28]
sub_hidden = [4,4]
sub_output = 10
# Evolution and population parameters
pop_key = 0
pop_size = 150
pop_elitism = 20
pop = Population(0, pop_size,pop_elitism)
num_generations = 10

def mnist(genomes):
	for genome_key, genome in genomes:
		cppn = CPPN.create(genome)
		substrate = decode(cppn,sub_input, sub_output, sub_hidden)

		fitness = 0
		for i in range(len(data_list)):
			correct_label = int(data_list[i][0])
			image_values = [(int(x)/255.0*0.99)+0.01 for x in data_list[i].split(',')]
			image_values = image_values[1:]
			actual_label = substrate.activate(image_values)
			actual_label = actual_label.index(max(actual_label))
			fitness += 1/10.0 if actual_label == correct_label else 0
		genome.fitness = fitness

winner_genome = pop.run_without_speciation(mnist,num_generations)
print("Winner Genome: {0} with Fitness {1}".format(winner_genome.key, 
	  winner_genome.fitness))
print("Output Nodes: {}".format(winner_genome.output_keys))
for node in winner_genome.nodes.values():
	print("Node {0} of type {1}".format(node.key, node.activation))

# Convert genome into usable CPPN
cppn = CPPN.create(winner_genome)
# Decode CPPN into substrate
substrate = decode(cppn,sub_input, sub_output, sub_hidden)

for i in range(len(data_list)):
	correct_label = int(data_list[i][0])
	print("Input: {}".format(correct_label))
	image_values = [(int(x)/255.0*0.99)+0.01 for x in data_list[i].split(',')]
	image_values = image_values[1:]
	actual_label = substrate.activate(image_values)
	actual_label = actual_label.index(max(actual_label))
	print("Actual Output: {}".format(actual_label))

draw_net(cppn, filename="images/mnist_cppn")
draw_net(substrate, filename="images/mnist_substrate")