#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
import os

import pickle
import numpy as np

import libs	

from interface.abm_interface import do_ABM_tactical
from abm_strategic.interface_distance import trajectories_from_data
from abm_tactical.generate_temporary_points import compute_temporary_points
from abm_strategic.utilities import write_trajectories_for_tact, Paras, convert_trajectories
from libs.general_tools import write_on_file, stdout_redirected, counter, clock_time
from libs.tools_airports import numberize_nodes, numberize_trajs
from libs.efficiency import rectificate_trajectories_network_with_time_and_alt

main_dir = os.path.abspath(__file__)
main_dir = os.path.split(os.path.dirname(main_dir))[0]

result_dir = libs.paths.result_dir

def points_in_sectors(zone, all_shapes, clean_double=True):
	"""
	BROKEN, DO NOT USE !!!
	"""
	points = []
	for pouet in all_shapes.keys():
		if all_shapes[pouet]['is_sector'] and pouet[:4] == zone:
			points.append(list(all_shapes[pouet]['boundary'][0].representative_point().coords)[0])

	if clean_double:
		points = list(set(points))

	return points

def name_results_shocks_old(paras, **param):
	m1_file_name = param['m1_file_name'].split('/')[-1].split('.dat')[0]
	output_file = result_dir + '/trajectories/M3/' + m1_file_name + '_sigV' + str(paras['sig_V']) + '_t_w' + str(paras['t_w'])
	if paras['f_shocks']>0:
		output_file += '_shocks' + str(paras['f_shocks'])
	output_file += '_' + str(param['i']) + '.dat'
	#print output_file
	return output_file

def name_results_shocks(paras, **param):
	m1_file_name = param['m1_file_name'].split('/')[-1].split('.dat')[0]
	output_file = result_dir + '/trajectories/M3/' + m1_file_name + '_sigV' + str(paras['sig_V']) + '_t_w' + str(paras['t_w'])
	#if paras['f_shocks']>0:
	output_file += '_shocks' + str(paras['f_shocks'])
	output_file += '.dat'
	#print output_file
	return output_file

def sweep_paras_shocks(zone, paras, input_files, config_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, force=False, trajectories=None,\
	temp_config_dir=main_dir+'/abm_tactical/config_temp', **kwargs):

	"""
	Script which sweeps two levels of parameters for the tactical ABM. Config files have to be given externally.
	The script creates a temporary folder to store all config files and avoid interactions between scripts.
	"""

	# config_file=main_dir + '/abm_tactical/config/config.cfg',\
	# shock_file=main_dir + '/abm_tactical/config/shock_tmp.dat',\
	# bound_file=main_dir + '/abm_tactical/config/bound_latlon.dat',\
	# tmp_navpoints_file=main_dir + '/abm_tactical/config/temp_nvp.dat',

	#config_file_temp = temp_config_dir + '/config.cfg'

	os.system('mkdir -p ' + temp_config_dir)
	
	shock_file = temp_config_dir + '/shock_tmp.dat'
	bound_file = temp_config_dir + '/bound_latlon.dat'
	tmp_navpoints_file = temp_config_dir + '/temp_nvp.dat'
	
	if trajectories!=None:
		write_trajectories_for_tact(trajectories, fil=input_file) 

	# Boundaries 
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


	# Temporary navpoints
	if not os.path.exists(tmp_navpoints_file_zone) or 1:
		tmp_nvp = compute_temporary_points(50000, boundary, save_file=tmp_navpoints_file_zone)
	os.system('cp ' + tmp_navpoints_file_zone + ' ' + tmp_navpoints_file)

	# Points for shocks
	if not os.path.exists(shock_file_zone) or 1:
		#points = points_in_sectors(zone, all_shapes)
		points = tmp_nvp[:100]
		with open(shock_file_zone, 'w') as f:
			for x, y in points:
				f.write(str(x) + '\t' + str(y) + '\n')
	os.system('cp ' + shock_file_zone + ' ' + shock_file)

	# Copy config file in temporary folder
	os.system('cp ' + config_file + ' ' + temp_config_dir + '/config.cfg')
	config_file = temp_config_dir + '/config.cfg'

	files, logs = [], []
	# Iterations on every input files
	for input_file in input_files:
		print "Input file:", input_file
		name_par1, name_par2 = tuple(paras['paras_to_loop'])
		for par1 in paras[name_par1 + '_iter']:
			print name_par1, "=", par1
			paras.update(name_par1, par1, config_file=config_file)
			for par2 in paras[name_par2 + '_iter']:
				print name_par2, "=", par2
				paras.update(name_par2, par2, config_file=config_file)
				with clock_time():
					output_file = name_results_shocks(paras, m1_file_name=input_file, zone=zone)
					for i in range(paras['nsim']):
						#counter(i, n_iter, message="Doing iterations... ")
						files.append((input_file, output_file.split('.dat')[0] + '_' + str(i) + '.dat'))
					
					if not os.path.exists(output_file.split('.dat')[0] + '_' + str(paras['nsim']-1) + '.dat') or force:
						#logs.append(output_file.split('.dat')[0] + '.txt')
						
						with stdout_redirected(to=output_file.split('.dat')[0] + '.txt'):
							do_ABM_tactical(input_file, output_file, config_file, verbose=1, 
								shock_tmp=shock_file,
								bound_latlon=bound_file,
								temp_nvp=tmp_navpoints_file)
							raise Exception()
		print

	return files, logs

def sweep_paras_shocks_rectificated(zone, paras, trajectories, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone,\
	target_eff=0.99, n_iter_rect=1, starting_date=[2010, 5, 6, 0, 0, 0], **kwargs):

	"""
	Deprecated
	"""

	trajs, G = trajectories_from_data(zone='LIRR', fmt='(n, z), t', data_version=None, save_file=None)
	G = G.to_undirected()
	numberize_nodes(G)
	numberize_trajs(trajs, G.mapping, fmt='(n, z), t')
	files = []
	print
	for i in range(n_iter_rect):
		print "Iteration", i, "of rectification..."
		trajs_rec, eff, G, groups_rec = rectificate_trajectories_network_with_time_and_alt(trajs, target_eff, G, remove_nodes=True)
		trajs_rec = convert_trajectories(G, trajs_rec, fmt_in='(n, z), t', put_sectors=True, remove_flights_after_midnight=True, starting_date=starting_date, input_minutes=True)
		input_file_rec = input_file.split('.dat')[0] + '_rect' + str(target_eff) + '_' + str(i) + '.dat'
		files_s = sweep_paras_shocks(zone, paras, trajs_rec, input_file_rec, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, **kwargs)
		files += files_s
		print

	return files
	
def lifetime_func(time_shock, t_w, t_r, t_i):
	return time_shock/(t_w*t_r*t_i)

def Nm_shock_func(f_shocks, t_w, t_r, t_i, DAY, shock_f_lvl_min, shock_f_lvl_max):
	return f_shocks*t_w*t_r*t_i/(DAY*(shock_f_lvl_max-shock_f_lvl_min)/10.)

if __name__=='__main__':

	# Read parameters 
	temp = __import__(sys.argv[1], globals(), locals(), ['all_paras'], -1)
	all_paras = temp.all_paras
	args = all_paras['args']
	del all_paras['args']

	# Sweep parameters
	print
	# for f in all_paras['input_files']:
	# 	print "Input file:", f
	# 	args[2] = f
	files, logs = sweep_paras_shocks(*args, **all_paras)
	#	print

	# Build complete log
	log_tot = result_dir + '/trajectories/files/'+ sys.argv[1] + '_log.txt' 
	for f in logs:
		os.system('cat ' + f + ' >> ' + log_tot)
		os.system('rm ' + f)

	print "Log saved in", log_tot

	# Save list of M1 and M3 files
	list_files = result_dir + '/trajectories/files/' + sys.argv[1] + '_files.pic'
	with open(list_files, 'w') as f:
		pickle.dump(files, f)

	print "List of files saved in ",  list_files







    
