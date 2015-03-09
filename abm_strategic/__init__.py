#from abm_strategic import import_exterior_libs
import os, sys
main_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.insert(1, os.path.join(__file__, main_dir))
sys.path.insert(1, os.path.join(main_dir, 'libs'))

# Public interface
import paths
result_dir = paths.result_dir
from simulationO import generate_traffic, do_standard
from iter_simO import average_sim, iter_sim
from prepare_navpoint_network import prepare_hybrid_network
from efficiency import rectificate_trajectories_network_with_time#, iter_partial_rectification
from utilities import draw_network_map, Paras