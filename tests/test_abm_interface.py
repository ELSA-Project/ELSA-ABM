#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest

"""
TODO
"""

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