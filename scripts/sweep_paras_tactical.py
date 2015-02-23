#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
import os

import pickle

from interface.abm_interface import choose_paras, do_ABM_tactical
from abm_strategic.interface_distance import produce_M1_trajs_from_data
from abm_tactical.generate_temporary_points import compute_temporary_points

# def build_path():
# 	pass

# def do(key_paras=None, new_value=None, input_file='', build_path_function=None):
# 	choose_paras(key_paras, new_value)
# 	do_ABM_tactical(input_file, output_file)

def sweep_paras(zone, n_iter=1, data_version=None):
	#paras_iter = {'sig_V':[0., 0.1]}
	#paras = Paras({'sig_V':0.})
	main_dir = os.path.abspath(__file__)
	main_dir = os.path.split(os.path.dirname(main_dir))[0]

	choose_paras('nsim', 1)
	choose_paras('tmp_from_file', 1)

	config_file = main_dir + '/abm_tactical/config/config.cfg'
	#loop(paras_iter, paras_iter.keys(), paras, thing_to_do=do, paras=paras, build_pat=build_pat)
	
	input_file = main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'
	#input_file = main_dir + '/trajectories/M1/trajs_for_testing_10_sectors.dat'
	
	produce_M1_trajs_from_data(zone=zone, data_version=data_version, put_fake_sectors=True, save_file=input_file)

	with open(main_dir + '/libs/All_shapes_334.pic','r') as f:
		all_shapes = pickle.load(f)
	boundary = list(all_shapes[zone]['boundary'][0].exterior.coords)
	assert boundary[0]==boundary[-1]

	with open('../abm_tactical/config/bound_latlon.dat', 'w') as f:
		for x, y in boundary:
			f.write(str(x) + '\t' + str(y) + '\n')

	compute_temporary_points(50000, boundary)

	sig_V_iter = [0., 0.01] #[0.] + [10**(-float(i)) for i in range(5, -1, -1)]
	t_w_iter = [40, 80] #[40, 80, 120, 160, 240] # times 8 sec 
	for sig_V in sig_V_iter:
		print "sig_V=", sig_V
		choose_paras('sig_V', sig_V)
		for t_w in t_w_iter:
			print "t_w=", t_w
			choose_paras('t_w', t_w)

			for i in range(n_iter):
				output_file = main_dir + '/trajectories/M3/trajs_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '.dat'
				do_ABM_tactical(input_file, output_file, config_file, verbose=1)


if __name__=='__main__':
	sweep_paras('LFMM')



    
