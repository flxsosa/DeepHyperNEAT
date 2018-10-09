import csv
import matplotlib.pyplot as plt
import numpy as np
from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
from visualize import draw_net
import math
import time

def convert_images(x):
	for i in range(len(x)):
		correct_label = int(x[i][0])
		img_vals = [(int(k)/255.0*0.99)+0.01 for k in x[i].split(',')]
		img_vals = img_vals[1:]
		img_vals.append(0.1)
		x[i] = (correct_label, img_vals)
	return x

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

# open the CSV file and read its contents into a list
data_file = open("mnist_dataset/mnist_simple.csv", 'r')
data_list = data_file.readlines()
data_file.close()
data_list = convert_images(data_list)

# Substrate parameters
sub_input = [28,28]
sub_hidden = [28,28]
sub_output = 3
# Evolution and population parameters
pop_key = 0
pop_size = 150
pop_elitism = 20
pop = Population(0, pop_size,pop_elitism)
num_generations = 10000

def mnist(genomes):
	for genome_key, genome in genomes:
		cppn = CPPN.create(genome)
		substrate = decode(cppn,sub_input, sub_output, sub_hidden)
		scorecard = []
		sum_y = [0]*sub_output
		for observation in data_list:
			correct_label = observation[0]
			image_values = observation[1]
			y_ = [0]*sub_output
			y_[correct_label] = 1
			y = softmax(substrate.activate(image_values))
			output_label = np.argmax(y)
			scorecard.append(1 if output_label == correct_label else 0)
		print("Accuracy: {}".format((sum(scorecard)+0.0)/len(scorecard)))
		genome.fitness = (sum(scorecard)+0.0)/len(scorecard)

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
	y = softmax(substrate.activate(image_values))
	output_label = np.argmax(y)
	print("Actual Output: {}".format(output_label))
draw_net(cppn, filename="images/mnist_cppn")