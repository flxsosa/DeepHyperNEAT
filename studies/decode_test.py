'''
Contains functions for decoding a given CPPN into a Substrate.

Felix Sosa
'''
import numpy as np
import itertools as it
from activations import ActivationFunctionSet
from phenomes import FeedForwardSubstrateWithBias
import time
# import numba
from multiprocessing import Process

def decode(cppn, sub_in_dims, sub_outputs, sheet_dims=None):
    # Input layer coordinate map
    x = np.linspace(-1.0, 1.0, sub_in_dims[1]) if (sub_in_dims[1] > 1) else [0.0]
    y = np.linspace(-1.0, 1.0, sub_in_dims[0]) if (sub_in_dims[0] > 1) else [0.0]
    sub_input_layer = list(it.product(x,y))
    # Output layer coordinate map
    x = np.linspace(-1.0,1.0,sub_outputs) if sub_outputs > 1 else [0.0]
    y = [0.0]
    sub_out_layer = list(it.product(x,y))
    # Check if sheet dimensions have been provided
    if sheet_dims:
        x = np.linspace(-1.0, 1.0, sheet_dims[1]) if (sheet_dims[1] > 1) else [0.0]
        y = np.linspace(-1.0, 1.0, sheet_dims[0]) if (sheet_dims[0] > 1) else [0.0]
        sheet = list(it.product(x,y))
    else:
        sheet = sub_input_layer
    # List of connection mappings
    connection_mappings = [cppn.nodes[x].cppn_tuple for x in cppn.output_nodes
                           if cppn.nodes[x].cppn_tuple[0] != (1,1)]
    # Substrate representation (dictionary of sheets)
    hidden_sheets = {cppn.nodes[node].cppn_tuple[0] for node in cppn.output_nodes}
    substrate = {s:sheet for s in hidden_sheets}
    substrate[(1,0)] = sub_input_layer
    substrate[(0,0)] = sub_out_layer
    # substrate[(1,1)] = np.array([(0.0,0.0)])
    cppn_idx_dict = {cppn.nodes[idx].cppn_tuple:idx for idx in cppn.output_nodes}
    return create_substrate(cppn,substrate, connection_mappings, cppn_idx_dict)

def create_substrate(cppn, substrate, mappings, id_dict):
    weight_matrices = {}
    # print(substrate)

    # for mapping in mappings:
    mapping = ((1,0),(2,0))
    cppnon_id = id_dict[mapping]
    weight_matrices[mapping] = define_weight_matrix(cppn, mapping,cppnon_id,substrate)
        

def define_weight_matrix(cppn,mapping,cppnon_id,substrate,max_weight=5.0):
    # Size of matrix
    # matrix = np.zeros([len(substrate[mapping[0]]),len(substrate[mapping[1]])])
    # Iterator over coordinate pairs
    coordinates = it.product(substrate[mapping[0]],substrate[mapping[1]])
    # coordinate_idx = it.product(range(len(substrate[mapping[0]])),range(len(substrate[mapping[1]])))
    for source_coord, target_coord in coordinates:
        print("Source Coord: {}".format(source_coord))
        print("Target Coord: {}".format(target_coord))
        inputs = [target_coord[0], target_coord[1], source_coord[0], source_coord[1]]
        weight = cppn.activate(inputs)[cppnon_id]
        # # x,y = next(coordinate_idx)
        # if abs(weight) > 0.2 and abs(weight) < max_weight:
        #     # matrix[x][y] = weight
        #     pass
        # elif abs(weight) > max_weight:
        #     # matrix[x][y] = max_weight
        #     pass
        # else:
        #     pass
    return None
