#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Script generating networks based on ACCs.
"""
import sys
sys.path.insert(1, '..')
import _mysql
import pickle
import numpy as np
from copy import deepcopy

import abm_strategic
from abm_strategic.simulationO import generate_traffic, write_down_capacities
from abm_strategic.interface_distance import name as name_net, paras_strategic
from abm_strategic.utilities import network_whose_name_is
from libs.tools_airports import my_conv
from libs.general_tools import silence, counter

result_dir = abm_strategic.result_dir

def name_sim(name_G):
	return  'trajs_' + name_G

def find_good_scaling_capacity(G, file_traffic, silent=True, max_value=1., min_value=0., step=-0.1, target=0.015, **paras_control):
	caps = np.arange(max_value, min_value, step)
	last_ratio = None
	for i, cap in enumerate(caps):
		with silence(silent):
			H = deepcopy(G)
			trajs, stats = generate_traffic(H, file_traffic=file_traffic, capacity_factor=cap, put_sectors=True,
										remove_flights_after_midnight=True, **paras_control)
			ratio = stats['rejected_flights']/float(stats['flights'])
		#print "cap=", cap, "; ratio=", ratio
		if ratio>=target:
			break
		last_ratio = ratio

	if ratio!=last_ratio and i!=0:
		#cap = (cap + caps[i-1])/2.
		cap = cap - (ratio - target)*(cap - caps[i-1])/(ratio - last_ratio)

	return cap, stats['rejected_flights']/float(stats['flights']), H

if __name__=='__main__':
	airac = 334
	starting_date=[2010,5,6]
	n_days=1
	cut_alt=240.
	mode='navpoints'
	data_version=None
	n_iter = 100
	target_rejected_flights = 0.015
	
	for country in ['LIRR']:
		# paras = paras_strategic(zone=country, airac=airac, starting_date=starting_date, n_days=n_days, cut_alt=cut_alt,\
		# 	mode=mode, data_version=data_version)

		# db=_mysql.connect("localhost","root", paras['password_db'], "ElsaDB_A" + str(airac), conv=my_conv)

		# db.query("""SELECT accName FROM ACCBoundary WHERE accName LIKE '""" + country + """%'""")
		# r=db.store_result()
		# rr=[rrr[0] for rrr in r.fetch_row(maxrows=0,how=0)]
		# db.close()
		for ACtot in [100, 500, 1000, 1500, 2000, 2500, 3000, 4000]:
			print "=============================================="
			print "    Generating traffic for ACtot:", ACtot
			print "=============================================="
			paras_nav = paras_strategic(zone=country, mode='navpoints', data_version=data_version) #TODO
			name_G = name_net(paras_nav, data_version)
			try:
				G = network_whose_name_is(result_dir + '/networks/' + name_G)
			except IOError:
				print "Could not load the network, I skip it."
				print 
				continue

			with open('../libs/All_shapes_334.pic','r') as f:
				all_shapes = pickle.load(f)
			boundary = list(all_shapes[country]['boundary'][0].exterior.coords)
			assert boundary[0]==boundary[-1]

			with open(result_dir + '/trajectories/bounds/' + G.name + '_bound_latlon.dat', 'w') as f:
				for x, y in boundary:
					f.write(str(x) + '\t' + str(y) + '\n')

			# --------------------------------------------------------------------------------------

			paras_control = {'bootstrap_mode':True, 'bootstrap_only_time':True, 'ACtot':ACtot}
			#print "Finding best capacity factor..."
			#capacity_factor, rejected_flights, H = find_good_scaling_capacity(G, result_dir + "/networks/" + name_G + '_flights_selected.pic', 
			#	target=target_rejected_flights,  silent=True, **paras_control)
			#print "Found best capacity factor:", capacity_factor, "(rejected fraction", rejected_flights, "of flights)"
			
			write_down_capacities(G, save_file=result_dir + '/trajectories/capacities/' + G.name + '_capacities_no_target_rej.dat')
			
			for i in range(n_iter):
				counter(i, n_iter, message="Doing simulations...")
				name_results = name_sim(name_G) + '_rej' + str(target_rejected_flights) + '_ACtot' + str(ACtot) + '_' + str(i) + '.dat'
				
				with silence(True):

					trajs, stats = generate_traffic(deepcopy(G), save_file= result_dir + '/trajectories/M1/' + name_results,
										record_stats_file=result_dir + '/trajectories/M1/' + name_results.split('.dat')[0] + '_stats.dat',
										file_traffic=result_dir + "/networks/" + name_G + '_flights_selected.pic',
										put_sectors=True,
										remove_flights_after_midnight=True,
										capacity_factor=1.,
										**paras_control)
				#print "Ratio rejected:", stats['rejected_flights']/float(stats['flights'])
		
			print 
			print 