#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')

#from abm_strategic.iter_simO import loop
#from abm.abm_strategic.utilities import Paras
from interface.abm_interface import choose_paras, do_ABM_tactical
from abm_strategic.interface_distance import produce_M1_trajs_from_data

# def build_path():
# 	pass

# def do(key_paras=None, new_value=None, input_file='', build_path_function=None):
# 	choose_paras(key_paras, new_value)
# 	do_ABM_tactical(input_file, output_file)

def sweep_paras(zone, date_version=None):
	#paras_iter = {'sig_V':[0., 0.1]}
	#paras = Paras({'sig_V':0.})

	#loop(paras_iter, paras_iter.keys(), paras, thing_to_do=do, paras=paras, build_pat=build_pat)
	input_file = '../trajectories/M1/trajs_LFRR_real_data.dat'
	produce_M1_trajs_from_data(zone=zone, data_version=data_version, save_file=input_file)

	with open('../libs/All_shapes_334.pic','r') as f:
		all_shapes = pickle.load(f)
	boundary = list(all_shapes[zone]['boundary'][0].exterior.coords)
	assert boundary[0]==boundary[-1]

	with open('../abm_tactical/config/bound_latlon.dat', 'w') as f:
		for x, y in boundary:
			f.write(str(x) + '\t' + str(y) + '\n')
	

	produce_M1_trajs_from_data 
	sig_V_iter = [0.] + [10**(-float(i)) for i in range(5, -1, -1)]
	look_ahead_iter = [0., 1.]
	for sig_V in sig_V_iter:
		choose_paras('sig_V', sig_V)
		for look_ahead in look_ahead_iter:
			# Put change of look-ahead.

			# Path 
			name_output = '../trajectories/M3/trajs_LFRR_real_data_sigV' + str(sig_V) + '_lkahd' + str(look_ahead) + '.dat'
			do_ABM_tactical(input_file, output_file)


if __name__=='__main__':
	sweep_paras()



    
