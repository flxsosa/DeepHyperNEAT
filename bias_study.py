from genome import Genome
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
from visualize import draw_net

sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1

# Make Genome
genome = Genome(1)

# Before
print("Before")
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print("Activation: {}\n".format(substrate.activate([1,1,1])))
draw_net(cppn, filename="reports/bias_cppn_before")
draw_net(substrate, filename="reports/bias_substrate_before")

# Depth
genome.mutate_increment_depth()
print("Depth")
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print("Activation: {}\n".format(substrate.activate([1,1,1])))
draw_net(cppn, filename="reports/bias_cppn_after_depth")
draw_net(substrate, filename="reports/bias_substrate_after_depth")

# DEpth 2
genome.mutate_increment_depth()
print("Depth 2")
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print("Activation: {}\n".format(substrate.activate([1,1,1])))
draw_net(cppn, filename="reports/bias_cppn_after_depth")
draw_net(substrate, filename="reports/bias_substrate_after_depth")

# Breadth
print("Breadth")
genome.mutate_increment_breadth()
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print("Activation: {}\n".format(substrate.activate([1,1,1])))
draw_net(cppn, filename="reports/bias_cppn_after_breadth")
draw_net(substrate, filename="reports/bias_substrate_after_breadth")

# Breadth 2
print("Breadth 2")
genome.mutate_increment_breadth()
cppn = CPPN.create(genome)
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
print("Activation: {}\n".format(substrate.activate([1,1,1])))
draw_net(cppn, filename="reports/bias_cppn_after_breadth")
draw_net(substrate, filename="reports/bias_substrate_after_breadth")