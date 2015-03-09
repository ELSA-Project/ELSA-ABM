#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic')

import os
from string import split
import numpy as np
import pickle
import random as rd

from abm_tactical import *
from abm_strategic import *

from libs.general_tools import silence

"""
TODO: put Paras Tact here, modify choose_paras to change several paras at the same time.
"""
# main_dir = os.path.abspath(__file__)
# main_dir = os.path.split(os.path.dirname(main_dir))[0]

# result_dir = os.path.join(os.path.dirname(main_dir), 'results')

if 0:
    # Manual seed
    see_=1
    #see_ = 15 #==> the one I used for new, old_good, etc.. 
    # see=2
    print "===================================="
    print "          USING SEED", see_
    print "===================================="
    np.random.seed(see_)

def choose_paras(name_para, new_value, fil="../abm_tactical/config/config.cfg"):
	"""
	Function to modify a config file for the tactical ABM. If the given parameters does not 
	exist in the config file, the function does not modify anything and exit silently.
	"""
	with open(fil) as f:
		lines = f.readlines()
	#print "Trying to set", name_para, "to new value", new_value
	new_lines = []
	for i, l in enumerate(lines):
		if l[0]!='#' and len(l)>1: # If not a comment and not a blank line
			value, name = l.strip('\n').split('\t#')#split(l.strip(), '\t#')
			if name == name_para:
				#print "found", name_para, "I put new value", new_value
				line = str(new_value) + "\t#" + name + '\n'*(line[-1]=='\n') # last bit because of shock_f_lvl_min
			else:
				line = l[:]
		else:
			line = l[:]
		new_lines.append(line)

	with open(fil, 'w') as f:
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

def do_ABM_tactical(input_file, output_file, config_file, verbose=2, 
	shock_tmp=main_dir + '/abm_tactical/config/shock_tmp.dat',
	bound_latlon= main_dir + '/abm_tactical/config/bound_latlon.dat',
	temp_nvp = main_dir + '/abm_tactical/config/temp_nvp.dat'):

	n = rd.randint(0, 1000000000)
	
	for fil, name in [(shock_tmp, 'shock_tmp'), (bound_latlon, 'bound_latlon'), (temp_nvp, 'temp_nvp')]:
		temp_file = fil + str(n)
		os.system('cp ' + fil + ' ' + temp_file)
		choose_paras(name, temp_file, fil=config_file)

	temp_config_file = config_file + str(n)
	os.system('cp ' + config_file + ' ' + temp_config_file)

	try:
		inpt = ["", input_file, output_file, temp_config_file]

		if verbose==2:
			print "M1 source:", inpt[1]
			print "Destination output:", inpt[2]
			print "Config file:", inpt[3]
			print
			print "Running ABM Tactical model..."

	
		with silence(verbose==0): # Does not work.
			tactical_simulation(inpt)
	finally:
		for fil, name in [(shock_tmp, 'shock_tmp'), (bound_latlon, 'bound_latlon'), (temp_nvp, 'temp_nvp')]:
			temp_file = fil + str(n)
			os.system('rm ' + temp_file)
			choose_paras(name, fil, fil=config_file)

		os.system('rm ' + temp_config_file)

	if verbose==2:
		print
		print "Done."

class ParasTact(Paras):
	def update(self, name_para, value_para, config_file=None):
		super(ParasTact, self).update(name_para, value_para)
		for k in self.to_update.keys() + self['paras_to_loop']:
			choose_paras(k, self[k], fil=config_file)

	def initialize_paras(self):
		"""
		This is here only because the update procedure does not
		compute the paras derived from other paras (via to_update)
		if the arguments of the functions have not changed. This is
		problematic for initialization because the derived paras
		might not be present in the dictionnary until one their 
		argument has changed.
		"""
		for key in self['paras_to_loop']:
			self[key] = self[key + '_iter'][0]

		keys = self.update_priority
		for key in keys:
			f, args = self.to_update[key]
			self[key] = f(*[self[arg] for arg in args])

if __name__ == '__main__':
	main_dir = os.path.abspath(__file__)
	main_dir = os.path.split(os.path.dirname(main_dir))[0]

	choose_paras('tmp_from_file', 1)
	#input_file = os.path.join(main_dir, "trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat")
	input_file = os.path.join(result_dir, "trajectories/M1/trajs_Real_LF_v5.8_Strong_EXTLFBB_LFBB_2010-5-6+0_d2_cut240.0_directed_1.dat")
	#output_file = os.path.join(main_dir, "results/output.dat")
	output_file = os.path.join(result_dir, "trajectories/M3/trajs_Real_LF_v5.8_Strong_EXTLFBB_LFBB_2010-5-6+0_d2_cut240.0_directed_1.dat")
	#do_ABM_tactical(input_file, output_file)
	do_efficiency2()

	#choose_paras('nsim', 10)


