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

from abm_strategic.simulationO import generate_traffic, write_down_capacities
from abm_strategic.interface_distance import name as name_net, paras_strategic
from abm_strategic.utilities import network_whose_name_is
from libs.tools_airports import my_conv
from libs.general_tools import silence, counter

def name_sim(name_G):
	return  'trajs_' + name_G

def find_good_scaling_capacity(G, file_traffic, max_value=1., min_value=0., step=-0.1, target=0.015):
	caps = np.arange(max_value, min_value, step)
	last_ratio = None
	for i, cap in enumerate(caps):
		with silence(True):
			H = deepcopy(G)
			trajs, stats = generate_traffic(H, file_traffic=file_traffic, capacity_factor=cap,put_sectors=True,
										remove_flights_after_midnight=True)
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
	target_rejected_flights = 0.

	targets_eff_per_ACC = {'EGTT':{0.954, 0.98, 0.999}, 
						   'LIMM':[0.954, 0.98, 0.999],
						   'LIRR':[0.98, 0.999],
						   'LFRR':[0.98, 0.999],
						   'LFFF':[0.98, 0.999],
						   'EGPX':[0.98, 0.999],
						   'LECP':[0.98, 0.999],
						   'LECM':[0.999],
						   'LECS':[0.999],
						   'LECB':[0.999],
						   'LFMM':[0.999],
						   'LFEE':[0.999],
						   'EGCC':[0.999],
						   'LIBB':[0.999],
						   'LIPP':[0.999],
						   'LFBB':[0.999]}

	
	for country in ['LF', 'LE', 'EG', 'EB', 'LI']:
		paras = paras_strategic(zone=country, airac=airac, starting_date=starting_date, n_days=n_days, cut_alt=cut_alt,\
			mode=mode, data_version=data_version)

		db=_mysql.connect("localhost","root", paras['password_db'], "ElsaDB_A" + str(airac), conv=my_conv)

		db.query("""SELECT accName FROM ACCBoundary WHERE accName LIKE '""" + country + """%'""")
		r=db.store_result()
		rr=[rrr[0] for rrr in r.fetch_row(maxrows=0,how=0)]
		db.close()

		for zone in rr:
			print "=============================================="
			print "    Generating traffic for zone:", zone
			print "=============================================="
			paras_nav = paras_strategic(zone=zone, mode='navpoints', data_version=data_version) #TODO
			name_G = name_net(paras_nav, data_version)
			try:
				G = network_whose_name_is('../networks/' + name_G)
			except IOError:
				print "Could not load the network, I skip it."
				print 
				continue

			with open('../libs/All_shapes_334.pic','r') as f:
				all_shapes = pickle.load(f)
			boundary = list(all_shapes[zone]['boundary'][0].exterior.coords)
			assert boundary[0]==boundary[-1]

			with open('../trajectories/bounds/' + G.name + '_bound_latlon.dat', 'w') as f:
				for x, y in boundary:
					f.write(str(x) + '\t' + str(y) + '\n')

			print "Finding best capacity factor..."
			capacity_factor, rejected_flights, H = find_good_scaling_capacity(G, "../networks/" + name_G + '_flights_selected.pic', target=target_rejected_flights)
			print "Found best capacity factor:", capacity_factor, "(rejected fraction", rejected_flights, "of flights)"
			
			write_down_capacities(H, save_file='../trajectories/capacities/' + G.name + '_capacities_rej' + str(target_rejected_flights) + '.dat')

			if zone in targets_eff_per_ACC.keys():
				for eff_target in targets_eff_per_ACC[zone]:
					for i in range(n_iter):
						counter(i, n_iter, message="Doing simulations...")
						name_results = name_sim(name_G) + '_eff_' + str(eff_target) + '_rej' + str(target_rejected_flights) + '_' + str(i) + '.dat'
						with silence(True):
							trajs, stats = generate_traffic(deepcopy(G), save_file='../trajectories/M1/' + name_results,
												record_stats_file='../trajectories/M1/' + name_results.split('.dat')[0] + '_stats.dat',
												file_traffic="../networks/" + name_G + '_flights_selected.pic',
												put_sectors=True,
												remove_flights_after_midnight=True,
												capacity_factor=capacity_factor,
												rectificate={'eff_target':eff_target, 'inplace':False, 'hard_fixed':False, 'remove_nodes':True, 'resample_trajectories':True}
												)

							#trajs_rec, eff, G, groups_rec = rectificate_trajectories_network(trajs, eff_target,	deepcopy(G), inplace=False)
					#print "Ratio rejected:", stats['rejected_flights']/float(stats['flights'])
		
			print 
			print 