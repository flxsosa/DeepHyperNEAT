'''
Contains classes for CPPN and Substrate phenomes.

Felix Sosa
'''

from util import itervalues
import numpy as np

def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False

def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required

def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers

class FeedForwardCPPN():
    def __init__(self, inputs, outputs, node_evals, nodes=None, mapping_tuples=None):
        '''
        Feed forward representation of a CPPN.

        inputs     -- input nodes of CPPN
        outpusts   -- output nodes of CPPN
        node_evals -- objects containing information for each node
        nodes      -- all nodes of CPPN
        mapping_tuples -- mapping tuples associated with each output node
        '''
        self.input_nodes = inputs
        self.output_nodes = {key:mapping_tuples[key] for key in mapping_tuples} if mapping_tuples else outputs
        self.node_evals = node_evals
        self.values = {key:0.0 for key in inputs + outputs}
        self.nodes = nodes
    
    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(
                                len(self.input_nodes), len(inputs)))
        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, incoming_connections in self.node_evals:
            node_inputs = []
            for node_id, conn_weight in incoming_connections:
                node_inputs.append(self.values[node_id] * conn_weight)
            s = agg_func(node_inputs)
            self.values[node] = act_func(s)
        return self.values

    @staticmethod
    def create(genome):
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
        layers = feed_forward_layers(genome.input_keys, genome.output_keys, connections)
        mapping_tuples = {}
        node_evals = []
        # Traverse layers
        for layer in layers:
            # For each node in each layer, collect all incoming connections to the node
            for node in layer:
                incoming_connections = []
                for conn_key in connections:
                    input_node, output_node = conn_key
                    if output_node == node:
                        cg = genome.connections[conn_key]
                        incoming_connections.append((input_node, cg.weight))
                # Gather node gene information
                node_gene = genome.nodes[node]
                activation_function = node_gene.activation
                node_evals.append((node, activation_function, sum, incoming_connections))
        # Gather mapping tuples
        for key in genome.output_keys:
            mapping_tuples[key] = genome.nodes[key].cppn_tuple
        for key in genome.bias_keys:
            mapping_tuples[key] = genome.nodes[key].cppn_tuple
        return FeedForwardCPPN(genome.input_keys, genome.output_keys, node_evals, genome.nodes, mapping_tuples)

class FeedForwardSubstrate():
    def __init__(self, inputs, bias, outputs, node_evals):
        self.input_nodes = inputs
        self.bias_node = bias
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):          
        if len(self.input_nodes+self.bias_node) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(
                                        len(self.input_nodes+self.bias_node), len(inputs)))
        for k, v in zip(self.input_nodes, inputs[:len(self.input_nodes)]):
            self.values[k] = v
        self.values[self.bias_node[0]] = inputs[-1]
        evaluations = self.node_evals[::-1]
        for node, act_func, agg_func, links in evaluations:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(s)
        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome):
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
        layers = feed_forward_layers(genome.input_keys, genome.output_keys, connections)
        node_evals = []
        # Traverse layers
        for layer in layers:
            # For each node in each layer, collect all incoming connections to the node
            for node in layer:
                inputs = []
                for conn_key in connections:
                    input_node, output_node = conn_key
                    if output_node == node:
                        cg = genome.connections[conn_key]
                        inputs.append((input_node, cg.weight))
                # Gather node gene information
                node_gene = genome.nodes[node]
                activation_function = node_gene.activation
                node_evals.append((node, activation_function, node_gene.bias, inputs))
        return FeedForwardCPPN(genome.input_keys, genome.bias_key, genome.output_keys, node_evals, genome.nodes)