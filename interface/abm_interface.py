#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic')

import os
from string import split
import numpy as np
import pickle

from abm_tactical import *
from abm_strategic import *
from abm_strategic.utilities import draw_network_map


if 0:
    # Manual seed
    see_=1
    #see_ = 15 #==> the one I used for new, old_good, etc.. 
    # see=2
    print "===================================="
    print "          USING SEED", see_
    print "===================================="
    np.random.seed(see_)

def choose_paras(name_para, new_value):
	with open("../abm_tactical/config/config.cfg") as f:
		lines = f.readlines()

	#print lines
	new_lines = []
	for i, l in enumerate(lines):
		#print i, l, len(l)
		if l[0]!='#' and len(l)>1: # If not a comment and not a blank line
			value, name = l.strip('\n').split('\t#')#split(l.strip(), '\t#')
			#name = 
			if name == name_para:
				line = str(new_value) + "\t#" + name + '\n'*(line[-1]=='\n') # last bit because of shock_f_lvl_min
			else:
				line = l[:]
		else:
			line = l[:]
		new_lines.append(line)
		#print line
	#print new_lines

	with open("../abm_tactical/config/config.cfg", 'w') as f:
		for line in new_lines:
			f.write(line)



def do_efficiency():
	# Choice of the network
	with open("../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2.pic", 'r') as f:
		G = pickle.load(f)

	# Choice of the trajectories
	trajectories, stats = generate_traffic(G, paras_file='../abm_strategic/paras.py', save_file=None, simple_setup=True, 
		starting_date=[2010, 6, 5, 10, 0, 0], coordinates=False, ACtot=3)

	geometrical_trajectories = list(zip(*trajectories)[0]) # without times of departure

	eff_targets = np.arange(0.9, 1., 0.01)
	final_trajs_list, final_eff_list, final_G_list, final_groups_list = iter_partial_rectification(geometrical_trajectories, eff_targets, G)

	for i, eff_target in enumerate(eff_targets):
		#traj_eff = rectificate_trajectories_network(trajectories, eff, G, groups=groups, probabilities=probabilities, remove_nodes = True)
		#final_trajs, final_eff, final_G, final_groups = partial_rectification(geometrical_trajectories, eff_target, G)
		draw_network_map(G.G_nav, title='Network map', trajectories=final_trajs_list[i], rep='./', airports=False, load=False, generated=True, add_to_title='', polygons=G.polygons.values(), numbers=False, show=False)
	draw_network_map(G.G_nav, title='Network map', trajectories=final_trajs_list[-1], rep='./', airports=False, load=False, generated=True, add_to_title='', polygons=G.polygons.values(), numbers=False, show=True)

def do_ABM_tactical(input_file, output_file):
	inpt = ["", input_file, output_file]

	print "M1 source:", inpt[1]
	print "Destination output:", inpt[2]
	print
	print "Running ABM Tactical model..."

	tactical_simulation(inpt)

	print
	print "Done."


if __name__ == '__main__':
	main_dir = os.path.abspath(__file__)
	main_dir = os.path.split(os.path.dirname(main_dir))[0]

	choose_paras('tmp_from_file', 1)
	#input_file = os.path.join(main_dir, "trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat")
	input_file = os.path.join(main_dir, "trajectories/M1/trajs_Real_LF_v5.8_Strong_EXTLFBB_LFBB_2010-5-6+0_d2_cut240.0_directed_1.dat")
	#output_file = os.path.join(main_dir, "results/output.dat")
	output_file = os.path.join(main_dir, "trajectories/M3/trajs_Real_LF_v5.8_Strong_EXTLFBB_LFBB_2010-5-6+0_d2_cut240.0_directed_1.dat")
	#do_ABM_tactical(input_file, output_file)
	do_efficiency()

	#choose_paras('nsim', 10)


