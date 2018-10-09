import csv

path = 'mnist_dataset/'
dataset = 'mnist_train_100.csv'
target = 'mnist_simple.csv'

new_dataset = open(path+target,'w')
old_dataset = open(path+dataset, 'r')

for line in old_dataset:
	if int(line[0]) in range(3):
		new_dataset.write(line)