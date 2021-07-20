'''
Contains functions for decoding a given CPPN into a Substrate.

create_substrate() and query_cppn() are based on corresponding
functions from PurePLES (but are heavily modified for DeepHyperNEAT).

Felix Sosa
'''
import numpy as np
import itertools as it
from deep_hyperneat.activations import ActivationFunctionSet
from deep_hyperneat.phenomes import FeedForwardSubstrate
import time

def decode(cppn, input_dimensions, output_dimensions, sheet_dimensions=None):
    '''
    Decodes a CPPN into a substrate.

    cppn             -- CPPN
    input_dimensions -- dimensions of substrate input layer
    output_dimension -- dimensions of substrate output layer
    sheet_dimensions -- optional substrate sheet dimensions
    '''

    # Create input layer coordinate map from specified input dimensions
    x = np.linspace(-1.0, 1.0, input_dimensions[1]) if (input_dimensions[1] > 1) else [0.0]
    y = np.linspace(-1.0, 1.0, input_dimensions[0]) if (input_dimensions[0] > 1) else [0.0]
    input_layer = list(it.product(x,y))

    # Create output layer coordinate map from specified output dimensions
    x = np.linspace(-1.0,1.0,output_dimensions) if (output_dimensions > 1) else [0.0]
    y = [0.0]
    output_layer = list(it.product(x,y))

    # Create sheet coordinate map from given sheet dimensions (if any)
    if sheet_dimensions:
        x = np.linspace(-1.0, 1.0, sheet_dimensions[1]) if (sheet_dimensions[1] > 1) else [0.0]
        y = np.linspace(-1.0, 1.0, sheet_dimensions[0]) if (sheet_dimensions[0] > 1) else [0.0]
        sheet = list(it.product(x,y))
    else:
        sheet = input_layer

    # Create list of mappings to be created between substrate sheets
    connection_mappings = [cppn.nodes[x].cppn_tuple for x in cppn.output_nodes if cppn.nodes[x].cppn_tuple[0] != (1,1)]

    # Create substrate representation (dictionary of sheets and their respective coordinate maps)
    hidden_sheets = {cppn.nodes[node].cppn_tuple[0] for node in cppn.output_nodes}
    substrate = {s:sheet for s in hidden_sheets}
    substrate[(1,0)] = input_layer
    substrate[(0,0)] = output_layer
    substrate[(1,1)] = [(0.0,0.0)]

    # Create dictionary of output node IDs to their respective mapping tuples
    cppn_idx_dict = {cppn.nodes[idx].cppn_tuple:idx for idx in cppn.output_nodes}

    # Create the substrate
    return create_substrate(cppn, substrate, connection_mappings, cppn_idx_dict)

def create_substrate(cppn, substrate, mapping_tuples, id_dict, act_func="relu"):
    '''
    Creates a neural network from a CPPN and substrate representation.

    Based on PurePLES. Copyright (c) 2017 Adrian Westh & Simon Krabbe Munck.

    cppn      -- CPPN
    substrate -- substrate representation (a dictionary of sheets and their respective coordinate maps)
    mapping_tuples -- list of mappings to be created between substrate sheets
    id_dict   -- dictionary of output node IDs and their respective mapping tuples
    act_func  -- optional argument for the activation function of the substrate
    '''
    node_evals, layers = [], gather_layers(substrate)

    # Assign coordinates to input, output, and bias layers
    input_coordinates, output_coordinates, bias_coordinates = (substrate[(1,0)],(1,0)), (substrate[(0,0)],(0,0)), (substrate[(1,1)],(1,1))

    # Assign ids to nodes in the substrate
    input_node_ids = list(range(len(input_coordinates[0])))
    bias_node_ids = list(range(len(input_node_ids), len(input_node_ids+bias_coordinates[0])))
    output_node_ids = list(range(len(input_node_ids+bias_node_ids), len(input_node_ids+bias_node_ids+output_coordinates[0])))

    # Remove the input and output layers from the substrate dictionary
    del substrate[(1,0)], substrate[(0,0)], substrate[(1,1)]

    # Create hidden layer coordinate maps
    hidden_coordinates = [(substrate[k], k) for k in substrate.keys()]

    # Assign ids to nodes in all hidden layers
    number_of_hidden_nodes = sum([len(layer[0]) for layer in hidden_coordinates])
    start_index = len(input_node_ids+output_node_ids+bias_node_ids)
    hidden_node_ids = list(range(start_index, start_index+number_of_hidden_nodes))

    # Get activation function for substrate
    act_func_set = ActivationFunctionSet()
    hidden_activation = act_func_set.get(act_func)
    output_activation = act_func_set.get('linear')

    # Decode depending on whether there are hidden layers or not
    if hidden_node_ids:

        # Query CPPN for mapping between output layer and topmost hidden layer
        out_hid_mapping_tuples = [mapping for mapping in mapping_tuples if mapping[1] == (0,0)]
        out_node_counter, idx, hidden_idx = 0, 0, 0
        # For each coordinate in output sheet
        for oc in output_coordinates[0]:
            # Adding Biases from Output to Hidden
            node_connections = query_cppn(cppn,oc,output_coordinates,bias_coordinates,bias_node_ids[0], id_dict)
            # For each connection mapping
            for mapping in out_hid_mapping_tuples:
                source_sheet_id = mapping[0]
                node_connections += query_cppn(cppn,oc,output_coordinates,(substrate[source_sheet_id],source_sheet_id), hidden_node_ids[idx], id_dict)
                idx += len(substrate[source_sheet_id])
            if node_connections:
                node_evals.append((output_node_ids[out_node_counter], output_activation, sum, node_connections))
            hidden_idx = idx
            idx = 0
            out_node_counter += 1

        # Query CPPN for mapping between hidden layers (from top to bottom)
        hid_node_counter = 0
        next_idx = idx = hidden_idx
        # For each hidden layer in the substrate, going from top to bottom
        for layer_idx in range((len(layers)-1), 2, -1):
            # For each sheet in the current layer, i
            for sheet_idx in range(len(layers[layer_idx])):
                # Assign target sheet id
                target_sheet_id = layers[layer_idx][sheet_idx]
                hid_hid_mapping_tuple = [mapping for mapping in mapping_tuples if (mapping[1] == target_sheet_id)]
                # For each coordinate in target sheet
                for hc in substrate[target_sheet_id]:
                    # Adding Biases from Hidden to Hidden
                    node_connections = query_cppn(cppn,hc,(substrate[target_sheet_id],target_sheet_id),bias_coordinates,bias_node_ids[0], id_dict)
                    for mapping in hid_hid_mapping_tuple:
                        source_sheet_id = mapping[0]
                        node_connections += query_cppn(cppn,hc,(substrate[target_sheet_id],target_sheet_id),(substrate[source_sheet_id], source_sheet_id),hidden_node_ids[idx], id_dict)
                        idx += len(substrate[source_sheet_id])
                    if node_connections:
                        node_evals.append((hidden_node_ids[hid_node_counter],hidden_activation,sum, node_connections))
                    hid_node_counter += 1
                    next_idx = idx
                    idx = hidden_idx
            idx = next_idx
            hidden_idx = next_idx

        # Query CPPN for mapping between bottom hidden layer to input layer
        idx = 0
        for i in range(len(layers[2])):
            # Assign target
            target_sheet_id = layers[2][i]
            # For each coordinate in target sheet
            for hc in substrate[target_sheet_id]:
                node_connections = query_cppn(cppn, hc, (substrate[target_sheet_id],target_sheet_id), input_coordinates, input_node_ids[idx], id_dict)
                # Adding Biases from Hidden to Input
                node_connections += query_cppn(cppn,hc,(substrate[target_sheet_id],target_sheet_id),bias_coordinates,bias_node_ids[0], id_dict)
                if node_connections:
                    node_evals.append((hidden_node_ids[hid_node_counter],hidden_activation, sum,node_connections))
                hid_node_counter += 1

    # No hidden layers
    else:
        # Output Input Layer
        idx, counter = 0, 0
        for i in range(len(layers[0])):
            # Assign target
            target_sheet_id = layers[0][i]
            # For each coordinate in target sheet
            for oc in output_coordinates[0]:
                node_connections = query_cppn(cppn,oc,output_coordinates,input_coordinates,input_node_ids[idx], id_dict)
                node_connections += query_cppn(cppn,oc,output_coordinates,bias_coordinates,bias_node_ids[idx], id_dict)
                if node_connections:
                    node_evals.append((output_node_ids[counter],output_activation,sum,node_connections))
                counter += 1

    return FeedForwardSubstrate(input_node_ids, bias_node_ids, output_node_ids, node_evals)

def query_cppn(cppn, source_coordinate, source_layer, target_layer, node_idx, id_dict, max_weight=5.0):
    '''
    Given a single node's coordinates and a layer of nodes, query the CPPN for potential weights
    for all possible connections between the layer and the single node.

    Based on PurePLES. Copyright (c) 2017 Adrian Westh & Simon Krabbe Munck.

    cppn         -- CPPN
    source_coordinate -- coordinate of single node to be connected to a set of nodes
    source_layer -- layer of nodes in which source_coordinate resides
    target_layer -- layer of nodes to which source_coordinate will be connected
    node_idx     -- node index to begin on when traversing target_layer
    id_dict      -- dictionary of CPPN output node ids and their respective mapping tuples
    '''
    node_connections = []
    target_coordinates = target_layer[0]
    target_layer_id = target_layer[1]
    source_layer_id = source_layer[1]
    mapping_tuple = (target_layer_id,source_layer_id)
    cppnon_id = id_dict[mapping_tuple]
    for target_coordinate in target_coordinates:
        i = [target_coordinate[0], target_coordinate[1], source_coordinate[0], source_coordinate[1]]
        w = cppn.activate(i)[cppnon_id]
        if abs(w) < max_weight:
            node_connections.append((node_idx, w*max_weight))
        elif abs(w) > max_weight:
            node_connections.append((node_idx, max_weight))
        else:
            node_connections.append((node_idx, 0.0))
        node_idx += 1
    return node_connections

def gather_layers(substrate):
    '''
    Takes a dictionary representation of a substrate and returns
    a list of the layers and the sheets within those layers.

    substrate -- dictionary representation of a substrate
    '''
    layers = {}
    for i in range(len(substrate)):
        layers[i] = []
        for key in substrate.keys():
            if key[0] == i and key not in layers[i]:
                layers[i].append(key)
        if layers[i] == []:
            del layers[i]
    return layers
