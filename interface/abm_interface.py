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

from libs.general_tools import silence


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

def do_efficiency2():
	# Choice of the network
	#name = "D_N44_nairports22_cap_constant_C5_w_coords_Nfp2"
	name = 'Real_LF_v5.8_Strong_EXTLF_2010-5-6+0_d2_cut240.0_directed'
	with open("../networks/" + name + ".pic", 'r') as f:
		G = pickle.load(f)

	# Choice of the trajectories
	trajectories, stats = generate_traffic(G, paras_file='../abm_strategic/paras.py', save_file=None, simple_setup=True, 
		starting_date=[2010, 6, 5, 10, 0, 0], coordinates=False, ACtot=3)

	geometrical_trajectories = list(zip(*trajectories)[0]) # without times of departure

	eff_targets = np.arange(0.90, 1.02, 0.02)
	final_trajs, final_eff, final_G, final_groups = rectificate_trajectories_network_with_time(trajectories, 0.97, G.G_nav, hard_fixed=False, resample_trajectories=False)
	geometrical_trajectories_final = list(zip(*final_trajs)[0])

	if 1:
		draw_network_map(G.G_nav, 	title='Network map', 
									trajectories=geometrical_trajectories, 
									figsize=(15, 10), 
									airports=False, 
									load=False, 
									flip_axes=True, 
									generated=True,
									polygons=G.polygons.values(), 
									numbers=False, 
									show=False,
									weight_scale=4.)
									#colors=colors,
									#sizes=sizes)

		# for i, eff_target in enumerate(eff_targets):
		# 	draw_network_map(final_G_list[i], title='Network map', 
		# 									  trajectories=final_trajs_list[i], 
		# 									  figsize=(15, 10), 
		# 									  airports=False, 
		# 									  load=False, 
		# 									  flip_axes=True, 
		# 									  generated=True, 
		# 									  add_to_title='', 
		# 									  polygons=G.polygons.values(), 
		# 									  numbers=False, 
		# 									  weight_scale=4.,
		# 									  show=(i==len(eff_targets)-1))

		draw_network_map(final_G, title='Network map', 
											  trajectories=geometrical_trajectories_final, 
											  figsize=(15, 10), 
											  airports=False, 
											  load=False, 
											  flip_axes=True, 
											  generated=True, 
											  add_to_title='', 
											  polygons=G.polygons.values(), 
											  numbers=False, 
											  weight_scale=4.,
											  show=True)
											  #colors=colors,
											  #sizes=sizes)

def do_efficiency():
	# TODO put this in test....
	# Choice of the network
	#name = "D_N44_nairports22_cap_constant_C5_w_coords_Nfp2"
	name = 'Real_LF_v5.8_Strong_EXTLF_2010-5-6+0_d2_cut240.0_directed'
	with open("../networks/" + name + ".pic", 'r') as f:
		G = pickle.load(f)

	# Choice of the trajectories
	trajectories, stats = generate_traffic(G, paras_file='../abm_strategic/paras.py', save_file=None, simple_setup=True, 
		starting_date=[2010, 6, 5, 10, 0, 0], coordinates=False, ACtot=1000)

	geometrical_trajectories = list(zip(*trajectories)[0]) # without times of departure

	eff_targets = np.arange(0.90, 1.02, 0.02)
	final_trajs_list, final_eff_list, final_G_list, final_groups_list, n_best = iter_partial_rectification(geometrical_trajectories, eff_targets, G, hard_fixed=True, resample_trajectories=False)

	if 1:
		colors, sizes = {}, {}
		for n in G.G_nav.nodes():
			if n in n_best:
				colors[n] = 'r'
				sizes[n] = 30
			else:
				colors[n] = 'b'
				sizes[n] = 20

		draw_network_map(G.G_nav, 	title='Network map', 
									trajectories=geometrical_trajectories, 
									figsize=(15, 10), 
									airports=False, 
									load=False, 
									flip_axes=True, 
									generated=True,
									polygons=G.polygons.values(), 
									numbers=False, 
									show=False,
									weight_scale=4.,
									colors=colors,
									sizes=sizes)

		# for i, eff_target in enumerate(eff_targets):
		# 	draw_network_map(final_G_list[i], title='Network map', 
		# 									  trajectories=final_trajs_list[i], 
		# 									  figsize=(15, 10), 
		# 									  airports=False, 
		# 									  load=False, 
		# 									  flip_axes=True, 
		# 									  generated=True, 
		# 									  add_to_title='', 
		# 									  polygons=G.polygons.values(), 
		# 									  numbers=False, 
		# 									  weight_scale=4.,
		# 									  show=(i==len(eff_targets)-1))

		draw_network_map(final_G_list[-1], title='Network map', 
											  trajectories=final_trajs_list[-1], 
											  figsize=(15, 10), 
											  airports=False, 
											  load=False, 
											  flip_axes=True, 
											  generated=True, 
											  add_to_title='', 
											  polygons=G.polygons.values(), 
											  numbers=False, 
											  weight_scale=4.,
											  show=True,
											  colors=colors,
											  sizes=sizes)

def do_ABM_tactical(input_file, output_file, config_file, verbose=2):
	inpt = ["", input_file, output_file, config_file]

	if verbose==2:
		print "M1 source:", inpt[1]
		print "Destination output:", inpt[2]
		print "Config file:", inpt[3]
		print
		print "Running ABM Tactical model..."

	with silence(verbose==0): # Does not work.
		tactical_simulation(inpt)

	if verbose==2:
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
	do_efficiency2()

	#choose_paras('nsim', 10)


