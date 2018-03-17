'''
Class for the Deep HyperNEAT default genome and genes.

Felix Sosa
'''
import numpy as np
from random import randint
from itertools import count
from six_util import iteritems,itervalues,iterkeys
from random import choice
from activations import ActivationFunctionSet
from copy import deepcopy

# Mutation probabilities
node_add_prob = 0.3
node_delete_prob = 0.2
conn_add_prob = 0.5
conn_delete_prob = 0.5
weight_mutation_rate = 0.8
bias_mutation_rate = 0.7
inc_depth_prob = 0.3
inc_breadth_prob = 0.2

class Genome():
	def __init__(self, key):
		# Unique genome identifier
		self.key = key
		self.node_indexer = None
		# (key, gene) pairs for gene sets
		self.connections = {}
		self.nodes = {}
		# Genome information
		self.fitness = None
		self.num_inputs = 4
		self.num_outputs = 1
		self.num_layers = 2
		self.input_keys = [-i - 1 for i in range(self.num_inputs)]
		self.output_keys = range(self.num_outputs)
		self.cppn_tuples = [((1,0), (0,0))]#[((1,0),(2,0)), ((2,0),(0,0))] #
		self.activations = ActivationFunctionSet()
		self.ancestors = []
		self.configure()
		self._complexity = len(self.nodes) + len(self.connections)

	def complexity(self):
		self._complexity = len(self.nodes) + len(self.connections)
		return self._complexity

	def configure(self):
		# Configure a new fully connected genome
		for input_id in self.input_keys:
			for output_id in self.output_keys:
				self.create_connection(input_id, output_id)
		for key, cppn_tuple in zip(self.output_keys,self.cppn_tuples):
			self.create_node('out',cppn_tuple,key)
	
	def copy(self, genome):
		# Copies the genes of another genome
		self.node_indexer = deepcopy(genome.node_indexer)
		self.num_inputs = deepcopy(genome.num_inputs)
		self.num_outputs = deepcopy(genome.num_outputs)
		self.input_keys = [x for x in genome.input_keys]
		self.output_keys = [x for x in genome.output_keys]
		self.cppn_tuples = [x for x in genome.cppn_tuples]
		self.num_layers = deepcopy(genome.num_layers)
		self.ancestors = deepcopy(genome.ancestors)
		# Nodes
		for node_copy in genome.nodes.values():
			node_to_add = NodeGene(node_copy.key,node_copy.type,
								   node_copy.activation, node_copy.cppn_tuple)
			node_to_add.bias = node_copy.bias
			self.nodes[node_to_add.key] = node_to_add
		# Connections
		for conn_copy in genome.connections.values():
			conn_to_add = ConnectionGene(conn_copy.key, conn_copy.weight)
			self.connections[conn_to_add.key] = conn_to_add

	def create_connection(self, source_key, target_key, weight=None):
		# Create a new connection gene
		if not weight:
			weight = np.random.uniform(-1,1)
		new_conn = ConnectionGene((source_key,target_key), weight)
		self.connections[new_conn.key] = new_conn
		return new_conn

	def create_node(self,node_type='hidden',cppn_tuple=None,key=None):
		# Create a new node
		activation_key = np.random.choice(self.activations.functions.keys())
		activation = self.activations.get(activation_key)
		new_node_key = self.get_new_node_key() if key == None else key
		new_node = NodeGene(new_node_key, node_type, activation, cppn_tuple)
		self.nodes[new_node.key] = new_node
		return new_node

	def mutate(self):
		# Mutate genome
		if np.random.uniform() < node_add_prob:
			self.mutate_add_node()
		if np.random.uniform() < node_delete_prob:
			self.mutate_delete_node()
		if np.random.uniform() < conn_add_prob:
			self.mutate_add_connection()
		if np.random.uniform() < conn_delete_prob:
			self.mutate_delete_connection()
		if np.random.uniform() < inc_depth_prob:
			self.mutate_increment_depth()
		if np.random.uniform() < inc_breadth_prob:
			self.mutate_increment_breadth()

		# Mutate connection genes.
		for conn_gene in self.connections.values():
			conn_gene.mutate(self)
		# Mutate node genes (bias, response, etc.).
		for node_gene in self.nodes.values():
			node_gene.mutate(self)

	def mutate_add_node(self):
		# Add new node to the genome
		# Choose connection to split
		if self.connections:
			idx = np.random.choice(range(len(self.connections)))
			conn_to_split = list(self.connections.keys())[idx]
		else:
			return
		# Create new hidden node and add to genome
		new_node = self.create_node()
		self.nodes[new_node.key] = new_node
		# Get weight from old connection
		old_weight = self.connections[conn_to_split].weight
		# Delete connection from genome
		del self.connections[conn_to_split]
		# Create i/o connections for new node
		i, o = conn_to_split
		self.create_connection(i, new_node.key, 1.0)
		self.create_connection(new_node.key, o, old_weight)
		self.ancestors.append("Mutation: Added Node {}".format(new_node.key))

	def mutate_add_connection(self):
		# Add a new connection to the genome
		# Gather possible target nodes and source nodes
		if not self.nodes:
			return
		possible_targets = list(iterkeys(self.nodes))
		target_key = choice(possible_targets)
		possible_sources = possible_targets + self.input_keys
		source_key = choice(possible_sources)
		# Ensure connection isn't duplicate
		if (source_key,target_key) in self.connections:
			self.connections[(source_key,target_key)].enabled = True
			return
		# Don't allow connections between two output nodes
		if source_key in self.output_keys and target_key in self.output_keys:
			return
		new_conn = self.create_connection(source_key, target_key)
		self.connections[new_conn.key] = new_conn
		self.ancestors.append("Mutation: Added Connection {} of weight {}".format(new_conn.key,new_conn.weight))

	def mutate_delete_node(self):
		# Delete a node
		available_nodes = [k for k in iterkeys(self.nodes) if k not in self.output_keys]
		if not available_nodes:
			return
		# Choose random node to delete
		del_key = np.random.choice(available_nodes)
		# Iterate through all connections and find connections to node
		conn_to_delete = set()
		for k, v in iteritems(self.connections):
			if del_key in v.key:
				conn_to_delete.add(v.key)
		for i in conn_to_delete:
			del self.connections[i]
		# Delete node key
		del self.nodes[del_key]
		self.ancestors.append("Mutation: Deleted Node {}".format(del_key))
		return del_key

	def mutate_delete_connection(self):
		# Delete a connection
		if self.connections:
			idx = np.random.choice(range(len(self.connections)))
			key = list(self.connections.keys())[idx]
			del self.connections[key]
			self.ancestors.append("Mutation: Deleted Connection {}".format(key))
	
	def mutate_increment_depth(self):
		# Add CPPNON to increment depth of Substrate
		# Create CPPN tuple
		source_layer = self.num_layers
		target_layer, target_sheet, source_sheet = 0, 0, 0
		cppn_tuple = ((source_layer, source_sheet),
					  (target_layer,target_sheet))
		# Adjust tuples for previous CPPNONs
		for key in self.output_keys:
			tup = self.nodes[key].cppn_tuple
			if tup[1] == (0,0):
				self.nodes[key].cppn_tuple = (tup[0], 
											  (source_layer,
											   source_sheet))
		# Create two new gaussian nodes
		gauss_1_node = self.create_node()
		gauss_1_node.activation = self.activations.get('dhngauss')
		gauss_1_node.bias = 0.0
		gauss_2_node = self.create_node()
		gauss_2_node.activation = self.activations.get('dhngauss')
		gauss_2_node.bias = 0.0
		gauss_3_node = self.create_node()
		gauss_3_node.activation = self.activations.get('dhngauss2')
		gauss_3_node.bias = 0.0
		# Create new CPPN Output Node (CPPNON)
		output_node = self.create_node('out', cppn_tuple)
		output_node.activation = self.activations.get('linear')
		output_node.bias = 0.0
		# Add new CPPNON key to list of output keys in genome
		self.num_outputs += 1
		self.num_layers += 1
		self.output_keys.append(output_node.key)
		# Add connections
		# x1 to gauss 1
		self.create_connection(self.input_keys[0], 
							gauss_1_node.key, -1.0)
		# x2 to gauss 1
		self.create_connection(self.input_keys[2], 
							gauss_1_node.key, 1.0)
		# y1 to gauss 2
		self.create_connection(self.input_keys[1], 
							gauss_2_node.key, -1.0)
		# y2 to gauss 2
		self.create_connection (self.input_keys[3], 
							gauss_2_node.key, 1.0) 
		# Gauss 1 to gauss 3
		self.create_connection(gauss_1_node.key, 
							gauss_3_node.key, 1.0)
		# Gauss 2 to gauss 3
		self.create_connection(gauss_2_node.key, 
							gauss_3_node.key, 1.0)
		# Gauss 3 to CPPNON
		self.create_connection(gauss_3_node.key,
							output_node.key,1.0)

	def mutate_increment_breadth(self):
		# Add CPPNON to increment breadth of Substrate
		# Can only expand a layer with more sheets if there is a hidden layer
		if self.num_layers <= 2:
			self.mutate_increment_depth()
		else:
			layer = randint(2,self.num_layers-1)
			# Find out how many sheets are represented by current CPPNONs
			num_sheets = len([x for x in self.output_keys if self.nodes[x].cppn_tuple[0][0] == layer])
			sheet = randint(0,num_sheets-1)
			copied_sheet = (layer, sheet)
			keys_to_append = []
			# Search for CPPNONs that contain the copied sheet
			for key in self.output_keys:
				# Create CPPNONs to represent outgoing connections
				if self.nodes[key].cppn_tuple[0] == copied_sheet:
					# create new cppn node for newly copied sheet
					cppn_tuple = ((layer,num_sheets),
								   self.nodes[key].cppn_tuple[1])
					output_node = self.create_node('out', cppn_tuple)
					output_node.activation = self.nodes[key].activation
					output_node.bias = self.nodes[key].bias
					keys_to_append.append(output_node.key)
					# Create connections in CPPN and halve existing connections
					for conn in list(self.connections):
						if conn[1] == key:
							self.connections[conn].weight /= 2.0
							self.create_connection(conn[0], output_node.key, 
												self.connections[conn].weight)

				# Create CPPNONs to represent the incoming connections
				if self.nodes[key].cppn_tuple[1] == copied_sheet:
					# create new cppn node for newly copied sheet
					cppn_tuple = (self.nodes[key].cppn_tuple[0],
								  (layer,num_sheets))
					output_node = self.create_node('out', cppn_tuple)
					output_node.activation = self.nodes[key].activation
					output_node.bias = self.nodes[key].bias
					keys_to_append.append(output_node.key)
					# Create connections in CPPN
					for conn in list(self.connections):
						if conn[1] == key:
							self.create_connection(conn[0], output_node.key, 
												self.connections[conn].weight)      
			# Add new CPPNONs to genome
			self.num_outputs += len(keys_to_append)
			self.output_keys.extend(keys_to_append)

	def mutate_add_mapping(self):
		# Adds connection mapping between two sheets previously
		# not connected
		layer_1 = 1#randint(1,self.num_layers-1)
		layer_2 = 0#randint(0,self.num_layers-1)
		num_sheets_1 = len([x for x in self.output_keys if self.nodes[x].cppn_tuple[0][0] == layer_1])
		num_sheets_2 = len([x for x in self.output_keys if self.nodes[x].cppn_tuple[0][0] == layer_2])
		sheet_1 = 0#randint(0,num_sheets_1-1)
		sheet_2 = 0 #if layer_2 == 0 else randint(0,num_sheets_2-1)

		source_sheet = (layer_1, sheet_1)
		target_sheet = (layer_2, sheet_2)

		cppn_tuple = (source_sheet, target_sheet)
		output_node = self.create_node('out', cppn_tuple)
		self.output_keys.append(output_node.key)
		print("Added node {} with tuple {}".format(output_node.key, output_node.cppn_tuple))
		for input_id in self.input_keys:
			self.create_connection(input_id, output_node.key)

	def get_new_node_key(self):
		# Returns new node key
		if self.node_indexer is None:
			self.node_indexer = count(max(self.output_keys)+1)
		new_id = next(self.node_indexer)
		assert new_id not in self.nodes
		return new_id

class NodeGene():
	def __init__(self,key,node_type,activation,cppn_tuple):
		self.type = node_type
		self.key = key
		self.bias = np.random.uniform(-1,1)
		self.activation = activation
		self.response = 1.0
		self.cppn_tuple = cppn_tuple

	def mutate(self,g):
		# Mutate attributes of node gene
		if np.random.uniform() < bias_mutation_rate:
			chg = np.random.uniform(-0.5,0.5)
			g.ancestors.append('Bias {} change from {} to {}'.format(self.key, self.bias, (self.bias+chg)))
			self.bias += chg

class ConnectionGene():
	def __init__(self,key,weight):
		self.key = key
		self.weight = weight
		self.enabled = True

	def mutate(self,g):
		# Mutate attributes of connection gene
		if np.random.uniform() < weight_mutation_rate:
			chg = np.random.uniform(-5,5)
			g.ancestors.append('Weight {} change from {} to {}'.format(self.key, self.weight, (self.weight+chg)))
			self.weight += chg
