from genome import Genome
from population import Population
from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
from visualize import draw_net

# Substrate Parameters
sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1

# Inputs
inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]

# Genome
genome = Genome(0)
# Genome, CPPN, and Substrate before mutation
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print("Before:",substrate.activate([1.0,2.0]))
print(substrate.values)

draw_net(cppn, filename="images/cppn_before_map")
draw_net(substrate, filename="images/substrate_before_map")

# Genome, CPPN, and Substrate after mutation
genome.mutate_increment_depth()
genome.mutate_increment_depth()
genome.mutate_add_mapping()
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print(substrate.node_evals)
print("After:",substrate.activate([1.0,2.0]))
print(substrate.values)

draw_net(cppn, filename="images/cppn_after_map")
draw_net(substrate, filename="images/substrate_after_map")