#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
import os

import pickle
import numpy as np

from interface.abm_interface import choose_paras, do_ABM_tactical
from abm_strategic.interface_distance import produce_M1_trajs_from_data
from abm_tactical.generate_temporary_points import compute_temporary_points
from libs.general_tools import write_on_file, stdout_redirected, counter

main_dir = os.path.abspath(__file__)
main_dir = os.path.split(os.path.dirname(main_dir))[0]

def points_in_sectors(zone, all_shapes, clean_double=True):
	points = []
	for pouet in all_shapes.keys():
		if all_shapes[pouet]['is_sector'] and pouet[:4] == zone:
			points.append(list(all_shapes[pouet]['boundary'][0].representative_point().coords)[0])

	if clean_double:
		points = list(set(points))

	return points

def name_results_shocks(**param):
	output_file = main_dir + '/trajectories/M3/trajs_' + param['zone'] + '_real_data_sigV' + str(param['sig_V']) + '_t_w' +\
		 str(param['t_w']) + '_shocks' + str(param['time_shock']) + '_' + str(param['i']) + '.dat'
	return output_file

def sweep_paras_shocks(zone, paras, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, n_iter=1, force=False,\
		config_file=main_dir + '/abm_tactical/config/config2.cfg',\
		shock_file=main_dir + '/abm_tactical/config/shock_tmp.dat',\
		bound_file=main_dir + '/abm_tactical/config/bound_latlon.dat',\
		tmp_navpoints_file=main_dir + '/abm_tactical/config/temp_nvp.dat'):
	
	with open(main_dir + '/libs/All_shapes_334.pic','r') as f:
		all_shapes = pickle.load(f)
	boundary = list(all_shapes[zone]['boundary'][0].exterior.coords)
	assert boundary[0]==boundary[-1]

	# Bounds
	if not os.path.exists(bound_file_zone):
		with open(bound_file_zone, 'w') as f:
			for x, y in boundary:
				f.write(str(x) + '\t' + str(y) + '\n')
	os.system('cp ' + bound_file_zone + ' ' + bound_file)


	# Points for shocks
	if not os.path.exists(shock_file_zone):
		points = points_in_sectors(zone, all_shapes)
		with open(shock_file_zone, 'w') as f:
			for x, y in points:
				f.write(str(x) + '\t' + str(y) + '\n')
	os.system('cp ' + shock_file_zone + ' ' + shock_file)

	# Temporary navpoints
	if not os.path.exists(tmp_navpoints_file_zone):
		compute_temporary_points(50000, boundary, save_file=tmp_navpoints_file_zone)
	os.system('cp ' + tmp_navpoints_file_zone + ' ' + tmp_navpoints_file)

	# Fixed parameters
	for key, value in paras.items():
		choose_paras(key, value, fil=config_file)

	time_shock = 60.*2. # in minutes
	f_shocks = 3. # Number of shocks per day.
	
	# Parameters to sweep
	#sig_V_iter = [0.] + [10**(-float(i)) for i in range(5, -1, -1)]
	sig_V_iter = np.arange(0., 0.26, 0.04)
	#sig_V_iter = [10**(-float(i)) for i in range(4, -1, -1)]
	#sig_V_iter = [0., 0.0001] # [0.] + [10**(-float(i)) for i in range(5, -1, -1)]
	#t_w_iter = [40, 80, 120, 160, 240] # times 8 sec 
	t_w_iter = [40, 60, 80, 100, 120]#, 160, 240] # times 8 sec 
	#t_w_iter = [40, 80] # [40, 80, 120, 160, 240] # times 8 sec 
	print 
	for sig_V in sig_V_iter:
		print "sig_V=", sig_V
		choose_paras('sig_V', sig_V, fil=config_file)
		for t_w in t_w_iter:
			print "t_w=", t_w
			lifetime = time_shock/(t_w*paras['t_r']*paras['t_i'])
			Nm_shock = f_shocks*t_w*paras['t_r']*paras['t_i']/(paras['DAY']*(paras['shock_f_lvl_max']-paras['shock_f_lvl_min'])/10.)
			print 'Nm_shock=', Nm_shock
			choose_paras('t_w', t_w, fil=config_file)
			choose_paras('lifetime', lifetime, fil=config_file)
			choose_paras('Nm_shock', Nm_shock, fil=config_file)

			for i in range(n_iter):
				counter(i, n_iter, message="Doing iterations... ")
				output_file = name_results_shocks(zone=zone, sig_V=sig_V, t_w=t_w, time_shock=time_shock, i=i)
				if not os.path.exists(output_file.split('.dat')[0] + '_0.dat') or force:
					with stdout_redirected(to=output_file.split('.dat')[0] + '.txt'):
						do_ABM_tactical(input_file, output_file, config_file, verbose=1)
		print

# def sweep_paras_shocks_rectificated(zone, paras, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, n_iter_rect=10, **kwargs):

	
# 	sweep_paras_shocks(zone, paras, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, **kwargs)
	
if __name__=='__main__':
	#main_dir = os.path.abspath(__file__)
	#main_dir = os.path.split(os.path.dirname(main_dir))[0]

	zone = 'LIRR'
	# Parameters directly controlled by tactical ABM
	paras = {}
	paras['nsim'] = 1
	paras['tmp_from_file'] = 1
	paras['DAY'] = 86400.
	paras['shock_f_lvl_min'] = 240
	paras['shock_f_lvl_max'] = 340
	paras['t_i'] = 8
	paras['t_r'] = 0.5

	# time_shock = 60.*2. # in minutes
	# f_shocks = 2. # Number of shocks per day.


	input_file=main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'
	shock_file_zone=main_dir + '/abm_tactical/config/shock_tmp_' + zone + '.dat'
	bound_file_zone= main_dir + '/abm_tactical/config/bound_latlon_' + zone + '.dat'
	tmp_navpoints_file_zone = main_dir + '/abm_tactical/config/temp_nvp_' + zone + '.dat'

	# M1 trajectories
	produce_M1_trajs_from_data(zone=zone, data_version=None, put_fake_sectors=True, save_file=input_file)

	sweep_paras_shocks(zone, paras, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, n_iter=100, force=True,
		config_file=main_dir + '/abm_tactical/config/config2.cfg')



    
