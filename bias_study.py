from genome import Genome
from phenomes import FeedForwardCPPN as CPPN 
from decode import decode
from visualize import draw_net

sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1

# Make Genome
genome = Genome(1)
# Make CPPN
cppn = CPPN.create(genome)
# Make Substrate
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
for n in genome.output_keys:
	print("Nodes Before: {}".format(genome.nodes[n].cppn_tuple))
# Report
draw_net(cppn, filename="reports/bias_cppn")
draw_net(substrate, filename="reports/bias_substrate")
# Mutate
genome.mutate_increment_depth()
genome.mutate_increment_depth()
print("Depth \n")
for n in genome.output_keys:
	print("Nodes After: {} {}".format(genome.nodes[n].cppn_tuple, n))
print()
print("Breadth")
genome.mutate_increment_breadth()
for n in genome.output_keys:
	print("Nodes After: {} {}".format(genome.nodes[n].cppn_tuple, n))
for n in genome.connections:
	print("Connections After: {} {}".format(genome.connections[n], n))
# Make CPPN
cppn = CPPN.create(genome)
# Make Substrate
substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)
substrate.activate([1,1,1])

# Report
draw_net(cppn, filename="reports/bias_cppn_after")
draw_net(substrate, filename="reports/bias_substrate_after")