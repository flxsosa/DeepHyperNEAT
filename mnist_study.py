import csv
import matplotlib.pyplot as plt
import numpy as np
from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
from visualize import draw_net
import math
# open the CSV file and read its contents into a list
data_file = open("mnist_dataset/mnist_simple.csv", 'r')
data_list = data_file.readlines()
data_file.close()
# substrate parameters
sub_input = [28,28]
sub_hidden = [4,4]
sub_output = 6
# Evolution and population parameters
pop_key = 0
pop_size = 150
pop_elitism = 20
pop = Population(0, pop_size,pop_elitism)
num_generations = 10000

def softmax(x):
	x = np.array(x,dtype=np.float128)
	# print(np.exp(x)/np.sum(np.exp(x)))
	return np.exp(x)/np.sum(np.exp(x))

def log_(x):
	for i in range(len(x)):
		x[i] = max(x[i],1e-15)
	return np.log(x)

def cross_entropy(y_, y):
	return -1*(y_*y)

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def binary_crossentropy(y_hat, y):
	return -1*(y*np.log(y_hat+0.0000001) + (1-y)*np.log(1-y_hat+0.0000001))

def mnist(genomes):
	for genome_key, genome in genomes:
		cppn = CPPN.create(genome)
		substrate = decode(cppn,sub_input, sub_output, sub_hidden)
		scorecard = []
		sum_y = [0]*sub_output
		for i in range(len(data_list)):
			correct_label = int(data_list[i][0])
			y_ = [0]*sub_output
			y_[correct_label] = 1
			print("Input: {}".format(y_))
			image_values = [(int(x)/255.0*0.99)+0.01 for x in data_list[i].split(',')]
			image_values = image_values[1:]
			image_values.append(0.1)
			y = softmax(substrate.activate(image_values))
			log_y = log_(y)
			cross_y = cross_entropy(y_,log_y)
			print("Actual Output: {}".format(y))
			sum_y += cross_y
		# print("Sum_Y: {}".format(sum_y))
		loss = np.mean(sum_y)
		print("Fitness: {}".format(loss))	
		genome.fitness = 1.0/loss

winner_genome = pop.run_with_speciation(mnist,num_generations)
print("Winner Genome: {0} with Fitness {1}".format(winner_genome.key, 
	  winner_genome.fitness))

# Convert genome into usable CPPN
cppn = CPPN.create(winner_genome)
substrate = decode(cppn,sub_input, sub_output, sub_hidden)
for i in range(len(data_list)):
	correct_label = int(data_list[i][0])
	print("Input: {}".format(correct_label))
	image_values = [(int(x)/255.0*0.99)+0.01 for x in data_list[i].split(',')]
	image_values = image_values[1:]
	image_values.append(0.1)
	actual_label = sigmoid(substrate.activate(image_values)[0])
	print("Actual Output: {}".format(actual_label))
draw_net(cppn, filename="images/mnist_cppn")
draw_net(substrate, filename="images/mnist_substrate")