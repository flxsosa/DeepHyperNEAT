'''
Class for the Deep HyperNEAT default genome and genes.

Felix Sosa
'''
import numpy as np
from itertools import count
from neat.six_util import iteritems,itervalues,iterkeys
from random import choice
from activations import ActivationFunctionSet

# Mutation probabilities
node_add_prob = 0.5
node_delete_prob = 0.1
conn_add_prob = 0.1
conn_delete_prob = 0.1
inc_depth_prob = 0
inc_breadth_prob = 0

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
		self.num_inputs = 5
		self.num_outputs = 2
		self.input_keys = [-i - 1 for i in range(self.num_inputs)]
		self.output_keys = [i for i in range(self.num_outputs)]
		self.cppn_tuples = [((1,0),(2,0)), ((2,0),(0,0))]
		self.activations = ActivationFunctionSet()
		self.configure()

	def configure(self):
		# Configure a new fully connected genome
		for input_id in self.input_keys:
			for outout_id in self.output_keys:
				self.create_connection(input_id, outout_id)
		
		for key, cppn_tuple in zip(self.output_keys,self.cppn_tuples):
			self.create_node('out',cppn_tuple,key)

	def create_connection(self, source_key, target_key, weight=None):
		# Create a new connection gene
		if not weight:
			weight = np.random.uniform(-5,5)
		new_conn = ConnectionGene((source_key,target_key), weight)
		self.connections[new_conn.key] = new_conn
		return new_conn

	def create_node(self,node_type='hidden',cppn_tuple=None,key=None):
		# Create a new node
		if node_type == 'hidden':
			activation_key = np.random.choice(self.activations.functions.keys())
			activation = self.activations.get(activation_key)
		else:
			activation = self.activations.get('linear')
		
		new_node_key = key if not key else self.get_new_node_key()
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
			conn_gene.mutate()
		# Mutate node genes (bias, response, etc.).
		for node_gene in self.nodes.values():
			node_gene.mutate()

	def mutate_add_node(self):
		# Add new node to the genome
		# Choose connection to split
		idx = np.random.choice(range(len(self.connections)))
		conn_to_split = list(self.connections.keys())[idx]
		# Create new hidden node and add to genome
		new_node = self.create_node()
		self.nodes[new_node.key] = new_node

		# Delete connection from genome
		del self.connections[conn_to_split]

		# Create i/o connections for new node
		i, o = conn_to_split
		self.create_connection(i, new_node.key, 1.0)
		self.create_connection(new_node.key, o, 1.0)

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

	def mutate_delete_node(self):
		# Delete a node
		available_nodes = list(iterkeys(self.nodes))
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
		return del_key

	def mutate_delete_connection(self):
		# Delete a connection
		if self.connections:
			idx = np.random.choice(range(len(self.connections)))
			key = list(self.connections.keys())[idx]
			del self.connections[key]

	def mutate_weight(self):
		# Perturbate weight
		pass
	
	def mutate_increment_depth(self):
		# Add CPPNON to increment depth of Substrate
		pass
	
	def mutate_incremrent_breadth(self):
		# Add CPPNON to increment breadth of Substrate
		pass
	
	def get_new_node_key(self):
		# Returns new node key
		if self.node_indexer is None:
			self.node_indexer = count(max(self.output_keys))
		new_id = next(self.node_indexer)
		assert new_id not in self.nodes
		return new_id

class NodeGene():

	def __init__(self,key,node_type,activation,cppn_tuple):
		self.type = node_type
		self.key = key
		self.bias = 0#np.random.uniform()
		self.activation = activation
		self.response = 1.0
		self.cppn_tuple = cppn_tuple

	def mutate(self):
		# Mutate attributes of node gene
		self.bias += np.random.uniform(-0.1,0.1)
		# self.response += np.random.uniform(-0.1,0.1)

class ConnectionGene():

	def __init__(self,key,weight):
		self.key = key
		self.weight = 0.0
		self.enabled = True

	def mutate(self):
		# Mutate attributes of connection gene
		self.weight += np.random.uniform(-1,1)