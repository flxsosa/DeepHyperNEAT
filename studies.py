from genome import Genome

from phenomes import FeedForwardCPPN as CPPN 
from phenomes import FeedForwardSubstrate as Substrate 
from decode import decode
from visualize import draw_net

# Substrate Parameters
sub_in_dims = [1,2]
sub_sh_dims = [1,1]
sub_o_dims = 1

genome = Genome(0)

cppn = CPPN.create(genome)

substrate = decode(cppn,sub_in_dims,sub_o_dims,sub_sh_dims)

draw_net(cppn, filename="images/cppn")
draw_net(substrate, filename="images/substrate")