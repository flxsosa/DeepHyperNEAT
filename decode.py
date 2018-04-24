'''
Contains functions for decoding a given CPPN into a Substrate.

Felix Sosa
'''
import numpy as np
import itertools as it
from activations import ActivationFunctionSet
from phenomes import FeedForwardSubstrateWithBias

def decode(cppn, sub_in_dims, sub_outputs, sheet_dims=None):
    # Decodes CPPN into a Substrate
    # Input layer coordinate map
    x = np.linspace(-1.0, 1.0, sub_in_dims[1]) if (sub_in_dims[1] 
                                                   > 1) else [0.0]
    y = np.linspace(-1.0, 1.0, sub_in_dims[0]) if (sub_in_dims[0] 
                                                   > 1) else [0.0]
    sub_input_layer = list(it.product(x,y))
    # Output layer coordinate map
    x = np.linspace(-1.0,1.0,sub_outputs) if sub_outputs > 1 else [0.0]
    y = [0.0]
    sub_out_layer = list(it.product(x,y))
    # Check if sheet dimensions have been provided
    if sheet_dims:
        x = np.linspace(-1.0, 1.0, sheet_dims[1]) if (sheet_dims[1] 
                                                      > 1) else [0.0]
        y = np.linspace(-1.0, 1.0, sheet_dims[0]) if (sheet_dims[0] 
                                                      > 1) else [0.0]
        sheet = list(it.product(x,y))
    else:
        sheet = sub_input_layer
    # List of connection mappings
    connection_mappings = [cppn.nodes[x].cppn_tuple for x in cppn.output_nodes
                           if cppn.nodes[x].cppn_tuple[0] != (1,1)]
    # Substrate representation (dictionary of sheets)
    hidden_sheets = {cppn.nodes[node].cppn_tuple[0] for node in 
                     cppn.output_nodes}
    substrate = {s:sheet for s in hidden_sheets}
    substrate[(1,0)] = sub_input_layer
    substrate[(0,0)] = sub_out_layer
    substrate[(1,1)] = [(0.0,0.0)]
    return create_phenotype_network(cppn,substrate, connection_mappings)

def create_phenotype_network(cppn, substrate, conn_maps, act_func="relu"):
    # Creates a neural network using a CPPN and Substrate representation
    connections, node_evals, layers = conn_maps, [], gather_layers(substrate)
    # Assign coordinates to input and output layers
    in_coords, out_coords = (substrate[(1,0)],(1,0)), (substrate[(0,0)],
                                                       (0,0))
    bias_coords = (substrate[(1,1)],(1,1))
    in_nodes = range(len(in_coords[0]))
    bias_nodes = range(len(in_nodes), len(in_nodes+bias_coords[0]))
    out_nodes = range(len(in_nodes+bias_nodes), len(in_nodes+bias_nodes+
                                                    out_coords[0]))
    # Remove the input and output layers from the substrate dictionary
    del substrate[(1,0)], substrate[(0,0)], substrate[(1,1)]
    # List of layers, first index = top layer.
    hid_coords = [(substrate[k], k) for k in substrate.keys()] 
    counter = sum([len(layer[0]) for layer in hid_coords])
    hid_idx = len(in_nodes+out_nodes+bias_nodes)
    hid_nodes = range(hid_idx, hid_idx+counter)
    # Get activation function for substrate
    act_func_set = ActivationFunctionSet()
    activation = act_func_set.get(act_func)
    out_activation = activation#act_func_set.get('linear')
    
    # Decode depending on whether there are hidden layers or not
    if hid_nodes:
        # Output to Topmost Hidden Layer
        conn_maps = [conn for conn in connections if conn[1] == (0,0)]
        counter, idx, hidden_idx = 0, 0, 0
        # For each coordinate in output sheet
        for oc in out_coords[0]:
            im = []
            # Adding Biases from Output to Hidden
            im += find_neurons(cppn,oc,out_coords[1],(bias_coords[0],
                               bias_coords[1]),bias_nodes[0], False)  
            # For each connection mapping
            for cm in conn_maps:
                src_sheet = cm[0]
                im += find_neurons(cppn,oc,out_coords[1],(substrate[src_sheet],
                                   src_sheet), hid_nodes[idx], False) 
                idx += len(substrate[src_sheet])
            if im: node_evals.append((out_nodes[counter], out_activation, 
                                      sum, 0.0, im))
            hidden_idx = idx
            idx = 0
            counter += 1
        
        # Hidden to Hidden Layers - from top to bottom
        counter = 0
        next_idx = idx = hidden_idx
        for i in range ((len(layers)-1), 2, -1):
            # For each sheet in the current layer, i
            for j in range(len(layers[i])):
                tgt_sheet = layers[i][j]
                conn_maps = [cm for cm in connections if (cm[1] == tgt_sheet)]
                # For each coordinate in target sheet
                for hc in substrate[tgt_sheet]:
                    # For each connection mapping
                    im = []
                    # Adding Biases from Hidden to Hidden
                    im += find_neurons(cppn,hc,tgt_sheet,(bias_coords[0], 
                                       bias_coords[1]),bias_nodes[0], False)  
                    for cm in conn_maps:
                        src_sheet = cm[0]
                        im += find_neurons(cppn,hc,tgt_sheet,(substrate[src_sheet], 
                                           src_sheet),hid_nodes[idx],False)     
                        idx += len(substrate[src_sheet])
                    if im: node_evals.append((hid_nodes[counter],activation,sum, 
                                              0.0,im))
                    counter += 1
                    next_idx = idx
                    idx = hidden_idx
            idx = next_idx
            hidden_idx = next_idx
        
        # Bottommost Hidden Layer to Input Layer
        idx = 0
        for i in range(len(layers[2])):
            # Assign target
            tgt_sheet = layers[2][i]
            # For each coordinate in target sheet
            for hc in substrate[tgt_sheet]:
                im = find_neurons(cppn, hc, tgt_sheet, (in_coords[0], 
                                  in_coords[1]), in_nodes[idx], False)
                # Adding Biases from Hidden to Input
                im += find_neurons(cppn,hc,tgt_sheet,(bias_coords[0], 
                                   bias_coords[1]),bias_nodes[0], False)
                if im: node_evals.append((hid_nodes[counter],activation, 
                                          sum,0.0,im))
                counter += 1

    else:
        # Output Input Layer
        idx, counter = 0, 0
        for i in range(len(layers[0])):
            # Assign target
            tgt_sheet = layers[0][i]
            # For each coordinate in target sheet
            for oc in out_coords[0]:
                im = find_neurons(cppn,oc,out_coords[1],(in_coords[0], 
                                  in_coords[1]),in_nodes[idx], False)
                im += find_neurons(cppn,oc,out_coords[1],(bias_coords[0], 
                                   bias_coords[1]),bias_nodes[idx], False)
                if im: node_evals.append((out_nodes[counter], 
                                          out_activation,sum,0.0,im))
                counter += 1
    return FeedForwardSubstrateWithBias(in_nodes, bias_nodes, out_nodes, 
                                        node_evals)

def find_neurons(cppn, src_coord, tgt_lyr, src_lyr, start_idx, outgoing, 
                 max_weight=5.0):
    # Finds neurons able to be connected to current neuron
    im = []
    idx = start_idx
    tgt_nodes, src_lyr = src_lyr[0], src_lyr[1]
    cppnon_tuple = (src_lyr, tgt_lyr)
    for tgt_coord in tgt_nodes:
        w = query_cppn(src_coord, tgt_coord, outgoing, cppn, cppnon_tuple,
                       max_weight)
        if w is not 0.0: im.append((idx, w))
        idx += 1
    return im

def query_cppn(coord1, coord2, outgoing, cppn, cppnon_tuple, max_weight=5.0):
    # Queries CPPN with two Substrate coordinates for a weight
    idx = None
    for node_id in cppn.output_nodes:
        if(cppn.nodes[node_id].cppn_tuple == cppnon_tuple):
            idx = node_id
    assert idx != None
    if outgoing:
        i = [coord1[0], coord1[1], coord2[0], coord2[1]]
    else:
        i = [coord2[0], coord2[1], coord1[0], coord1[1]]
    w = cppn.activate(i)[idx]
    if abs(w) > 0.2 and abs(w) < max_weight:
        return w# * max_weight
    elif abs(w) > max_weight:
        return max_weight
    else:
        return 0.0

def gather_layers(substrate):
    l = {}
    for i in range(len(substrate)):
        l[i] = []
        for key in substrate.keys():
            if key[0] == i and key not in l[i]: l[i].append(key)
        if l[i] == []: del l[i]
    return l