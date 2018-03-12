'''
Contains functions for decoding a given CPPN into a Substrate.

Felix Sosa
'''

import numpy as np
import itertools as it
import activations
from phenomes import FeedForwardSubstrate


def decode(cppn, sub_input_dimensions, sub_outputs, sheet_dimensions=None):
    # Get x and y coordinates for Substrate input layer and create layer
    if sub_input_dimensions[1] == 1:
        x = [0.0]
    elif sub_input_dimensions[1] > 1:
        x = np.linspace(-1.0, 1.0, sub_input_dimensions[1])

    if sub_input_dimensions[0] == 1:
        y = [0.0]
    elif sub_input_dimensions[0] > 1:
        y = np.linspace(-1.0, 1.0, sub_input_dimensions[0])
    sub_input_layer = list(it.product(x,y))

    if sub_outputs == 1:
        x = [0.0]
    elif sub_outputs > 1:
        x = np.linspace(-1.0,1.0,sub_outputs)
    y = [0.0]
    sub_out_layer = list(it.product(x,y))

    # Check if sheet dimensions have been provided
    if sheet_dimensions:
        if sheet_dimensions[1] == 1:
            x = [1.0]
        elif sheet_dimensions[1] > 1:
            x = np.linspace(-1.0, 1.0, sheet_dimensions[1])

        if sheet_dimensions[0] == 1:
            y = [0.0]
        elif sheet_dimensions[0] > 1:
            y = np.linspace(-1.0, 1.0, sheet_dimensions[0])
        sheet = list(it.product(x,y))
    else:
        sheet = sub_input_layer

    # Initialize substrate with input (1,0) and output (0,0) layers
    # Should do dict comprehension
    substrate = {
        (0,0): sub_out_layer,
        (1,0): sub_input_layer
    }
    # print(substrate)
    # Create a list of connection mappings
    connection_mappings = []

    # Traverse CPPN Output Nodes (CPPNONs) and add layers and sheets to the substrate
    for node in cppn.output_nodes:
        # Add cppn_tuple to connection mapping list
        connection_mappings.append(cppn.nodes[node].cppn_tuple)
        if cppn.nodes[node].cppn_tuple[0] not in substrate:
            substrate[cppn.nodes[node].cppn_tuple[0]] = sheet
        if cppn.nodes[node].cppn_tuple[1] not in substrate:
            substrate[cppn.nodes[node].cppn_tuple[1]] = sheet
    
    return create_phenotype_network(cppn,substrate, connection_mappings)

def create_phenotype_network(cppn, substrate, connection_maps, activation_function="relu"):
    '''
    Creates a NN using a CPPN
    '''
    connections = connection_maps
    layers = {}
    # Inititialize node evaluations
    node_evals = []
    # Gather layers in substrate
    for i in range(len(substrate.keys())):
        layers[i] = []
        for key in substrate.keys():
            if key[0] == i and key not in layers[i]:
                layers[i].append(key)
        if layers[i] == []:
            del layers[i]
    # Assign coordinates
    input_coordinates = (substrate[(1,0)],(1,0))
    input_nodes = range(len(input_coordinates[0]))
    output_coordinates = (substrate[(0,0)],(0,0))
    output_nodes = range(len(input_nodes), len(input_nodes)+len(output_coordinates[0]))
    # Remove the input and output layers from the substrate dictionary
    del substrate[(1,0)]
    del substrate[(0,0)]
    # List of layers, first index = top layer.
    hidden_coordinates = [(substrate[k], k) for k in substrate.keys()] 
    counter = 0
    for layer in hidden_coordinates:
        counter += len(layer[0])
    hidden_nodes = range(len(input_nodes)+len(output_nodes), 
                         len(input_nodes)+len(output_nodes)+counter)
    # Get activation function.
    activation_functions = activations.ActivationFunctionSet()
    activation = activation_functions.get(activation_function)
    # Decode depending on whether there are hidden layers or not
    if len(hidden_nodes) != 0:
        # Output to Topmost Hidden Layer
        # Find connection mappings with output layer as target sheet
        connection_mappings = [c for c in connections if c[1] == (0,0)]
        counter = 0
        idx = 0
        hidden_idx = 0
        # For each coordinate in output sheet
        for oc in output_coordinates[0]:
            im = []
            # For each connection mapping
            for cm in connection_mappings:
                source_sheet = cm[0]
                im += find_neurons(cppn, oc, output_coordinates[1], 
                                   (substrate[source_sheet], source_sheet), hidden_nodes[idx], False)  
                idx += len(substrate[source_sheet])
            if im:
                node_evals.append((output_nodes[counter], activation, sum, 0.0, 1.0, im))
            hidden_idx = idx
            idx = 0
            counter += 1
        # Hidden to Hidden Layers - from top to bottom
        # For each layer in substrate, omitting input layer and going from top to bottom
        counter = 0
        next_idx = idx = hidden_idx
        for i in range ((len(layers)-1), 2, -1):
            # For each sheet in the current layer, i
            for j in range(len(layers[i])):
                # Assign the target layer
                target_sheet = layers[i][j]
                # Find connection mappings with current sheet as target sheet
                connection_mappings = [cm for cm in connections if cm[1] == target_sheet]
                # For each coordinate in target sheet
                for hc in substrate[target_sheet]:
                    # For each connection mapping
                    im = []
                    for cm in connection_mappings:
                        source_sheet = cm[0]
                        im += find_neurons(cppn, hc, target_sheet, 
                                           (substrate[source_sheet], source_sheet), 
                                           hidden_nodes[idx], False)       
                        idx += len(substrate[source_sheet])
                    if im:
                        node_evals.append((hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
                    counter += 1
                    next_idx = idx
                    idx = hidden_idx
            idx = next_idx
            hidden_idx = next_idx
        
        # Bottommost Hidden Layer to Input Layer
        # For each sheet in second layer in substrate
        idx = 0
        for i in range(len(layers[2])):
            # Assign target
            target_sheet = layers[2][i]
            # For each coordinate in target sheet
            for hc in substrate[target_sheet]:
                im = find_neurons(cppn, hc, target_sheet, (input_coordinates[0], 
                                  input_coordinates[1]), input_nodes[idx], False)
                if im:
                    node_evals.append((hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
                counter += 1
    else:
        # Output Input Layer
        idx = 0
        counter = 0
        for i in range(len(layers[0])):
            # Assign target
            target_sheet = layers[0][i]
            # For each coordinate in target sheet
            for oc in output_coordinates[0]:
                im = find_neurons(cppn, oc, output_coordinates[1], (input_coordinates[0], 
                                  input_coordinates[1]), input_nodes[idx], False)
                if im:
                    node_evals.append((output_nodes[counter], activation, sum, 0.0, 1.0, im))
                counter += 1
    return FeedForwardSubstrate(input_nodes, output_nodes, node_evals)

def find_neurons(cppn, source_coord, source_layer, target_layer, start_idx, outgoing, max_weight=5.0):
    '''
    Helper function
    '''
    im = []
    idx = start_idx
    target_nodes = target_layer[0]
    target_layer = target_layer[1]
    cppnon_tuple = (target_layer, source_layer)
    for target_coord in target_nodes:
        w = query_cppn(source_coord, target_coord, outgoing, cppn, cppnon_tuple, max_weight)
        if w is not 0.0:  # Only include connection if the weight isn't 0.0.
            im.append((idx, w))
        idx += 1
    return im

def query_cppn(coord1, coord2, outgoing, cppn, cppnon_tuple, max_weight=5.0):
    '''
    Helper funciton. Queries CPPN for weights in a Substrate.
    '''
    idx = None
    # print([coord1[0], coord1[1], coord2[0], coord2[1]])
    for node_id in cppn.output_nodes:
        if(cppn.nodes[node_id].cppn_tuple == cppnon_tuple):
            idx = node_id
    if idx == None:
        return 0.0
    if outgoing:
        i = [coord1[0], coord1[1], coord2[0], coord2[1]]
    else:
        i = [coord2[0], coord2[1], coord1[0], coord1[1]]
    w = cppn.activate(i)[idx]
    if abs(w) > 0.2:  # If abs(weight) is below threshold, treat weight as 0.0.
        return w# * max_weight
    else:
        return 0.0