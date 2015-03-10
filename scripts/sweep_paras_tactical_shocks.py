#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
import os

import pickle
import numpy as np

from interface.abm_interface import choose_paras, do_ABM_tactical
from abm_strategic.interface_distance import produce_M1_trajs_from_data, trajectories_from_data
from abm_tactical.generate_temporary_points import compute_temporary_points
from abm_strategic.utilities import write_trajectories_for_tact, Paras, convert_trajectories
from libs.general_tools import write_on_file, stdout_redirected, counter
from libs.tools_airports import numberize_nodes, numberize_trajs
from libs.efficiency import rectificate_trajectories_network_with_time_and_alt

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

def name_results_shocks(paras, **param):
	m1_file_name = param['m1_file_name'].split('/')[-1].split('.dat')[0]
	output_file = main_dir + '/trajectories/M3/' + m1_file_name + '_sigV' + str(paras['sig_V']) + '_t_w' + str(paras['t_w'])
	if paras['f_shocks']>0:
		output_file += '_shocks' + str(paras['f_shocks'])
	output_file += '_' + str(param['i']) + '.dat'
	#print output_file
	return output_file

def sweep_paras_shocks(zone, paras, trajectories, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, n_iter=1, force=False,\
	config_file=main_dir + '/abm_tactical/config/config2.cfg',\
	shock_file=main_dir + '/abm_tactical/config/shock_tmp.dat',\
	bound_file=main_dir + '/abm_tactical/config/bound_latlon.dat',\
	tmp_navpoints_file=main_dir + '/abm_tactical/config/temp_nvp.dat'):
	
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

	# Iterations
	name_par1, name_par2 = tuple(paras['paras_to_loop'])
	for par1 in paras[name_par1 + '_iter']:
		print name_par1, "=", par1
		paras.update(name_par1, par1, config_file=config_file)
		for par2 in paras[name_par2 + '_iter']:
			print name_par2, "=", par2
			paras.update(name_par2, par2, config_file=config_file)

			for i in range(n_iter):
				counter(i, n_iter, message="Doing iterations... ")
				output_file = name_results_shocks(paras, m1_file_name=input_file, zone=zone, i=i)
				if not os.path.exists(output_file.split('.dat')[0] + '_0.dat') or force:
					with stdout_redirected(to=output_file.split('.dat')[0] + '.txt'):
						do_ABM_tactical(input_file, output_file, config_file, verbose=1)
		print

def sweep_paras_shocks_rectificated(zone, paras, trajectories, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone,\
	 target_eff=0.99, n_iter_rect=10, starting_date=[2010, 5, 6, 0, 0, 0], **kwargs):

	trajs, G = trajectories_from_data(zone='LIRR', fmt='(n, z), t', data_version=None, save_file=None)
	G = G.to_undirected()
	numberize_nodes(G)
	numberize_trajs(trajs, G.mapping, fmt='(n, z), t')
	for i in range(n_iter_rect):
		trajs_rec, eff, G, groups_rec = rectificate_trajectories_network_with_time_and_alt(trajs, target_eff, G, remove_nodes=True)
		trajs_rec = convert_trajectories(G, trajs_rec, fmt_in='(n, z), t', put_sectors=True, remove_flights_after_midnight=True, starting_date=starting_date)
		input_file_rec = input_file.split('.dat')[0] + '_rect' + str(target_eff) + '_' + str(i) + '.dat'
		sweep_paras_shocks(zone, paras, trajs_rec, input_file_rec, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, **kwargs)
	
def lifetime_func(time_shock, t_w, t_r, t_i):
	return time_shock/(t_w*t_r*t_i)

def Nm_shock_func(f_shocks, t_w, t_r, t_i, DAY, shock_f_lvl_min, shock_f_lvl_max):
	return f_shocks*t_w*t_r*t_i/(DAY*(shock_f_lvl_max-shock_f_lvl_min)/10.)

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
		for key in paras['paras_to_loop']:
			paras[key] = paras[key + '_iter'][0]

		keys = self.update_priority
		for key in keys:
			f, args = self.to_update[key]
			self[key] = f(*[self[arg] for arg in args])

if __name__=='__main__':
	#main_dir = os.path.abspath(__file__)
	#main_dir = os.path.split(os.path.dirname(main_dir))[0]

	zone = 'LIRR'
	# Parameters directly controlled by tactical ABM
	paras = ParasTact({})
	paras['nsim'] = 1
	paras['tmp_from_file'] = 1
	paras['DAY'] = 86400.
	paras['shock_f_lvl_min'] = 240
	paras['shock_f_lvl_max'] = 340
	paras['t_i'] = 8
	paras['t_r'] = 0.5
	paras['time_shock'] = 60.*2. # in minutes
	paras['f_shocks'] = 3. # Number of shocks per day.

	paras['sig_V_iter'] = np.arange(0., 0.26, 0.04)
	paras['t_w_iter'] = [40, 60, 80, 100, 120]

	paras['paras_to_loop'] = ['sig_V', 't_w']

	assert len(paras['paras_to_loop'])==2

	paras.to_update['lifetime'] = (lifetime_func, ('time_shock', 't_w', 't_r', 't_i'))
	paras.to_update['Nm_shock'] = (Nm_shock_func, ('f_shocks', 't_w', 't_r', 't_i', 'DAY', 'shock_f_lvl_min', 'shock_f_lvl_max'))

	paras.update_priority = ['lifetime', 'Nm_shock']

	paras.analyse_dependance()
	paras.initialize_paras()

	input_file=main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'
	shock_file_zone=main_dir + '/abm_tactical/config/shock_tmp_' + zone + '.dat'
	bound_file_zone= main_dir + '/abm_tactical/config/bound_latlon_' + zone + '.dat'
	tmp_navpoints_file_zone = main_dir + '/abm_tactical/config/temp_nvp_' + zone + '.dat'

	# M1 trajectories
	trajectories = produce_M1_trajs_from_data(zone=zone, data_version=None, put_fake_sectors=True)#, save_file=input_file)


	#sweep_paras_shocks(zone, paras, trajectories, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, n_iter=100, force=False,
	#	config_file=main_dir + '/abm_tactical/config/config2.cfg')

	sweep_paras_shocks_rectificated(zone, paras, trajectories, input_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone, starting_date=[2010, 5, 6, 0, 0, 0])




    
