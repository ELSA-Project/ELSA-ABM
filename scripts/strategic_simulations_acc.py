#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Script generating simulations for different ACCs.
"""
import sys
sys.path.insert(1, '..')
import _mysql
import pickle
import numpy as np
from copy import deepcopy

from libs.paths import result_dir as _result_dir
from abm_strategic.simulationO import generate_traffic, write_down_capacities
from abm_strategic.interface_distance import name as name_net, paras_strategic
from abm_strategic.utilities import network_whose_name_is
from libs.tools_airports import my_conv
from libs.general_tools import silence, counter

def name_sim(name_G):
	return  'trajs_' + name_G

def find_good_scaling_capacity(G, file_traffic, max_value=1., min_value=0., step=-0.1, target=0.015):
	"""
	This function scales down uniformly the capacities of all sectors of G 
	in order to meet the target of fraction of flights rejected.

	Parameters
	----------
	G: hybrid network
	file_traffic: string
		full path of file with traffic to use to infer capacities
	max_value: float, optional
		maximum scale value.
	min_value: float, optional
		minimum scale value
	step: float, optional
		step of scale value for the search.
	target: float, optional
		Target of fraction of flight rejected.

	Returns
	-------
	cap: float
		scale factor which meets the target
	frac: float
		exact fraction of flights rejected with this target
	H: hybrid network
		with modified value of capacities.
	
	"""
	
	caps = np.arange(max_value, min_value, step)
	last_ratio = None

	# Sweep the scaling factor
	for i, cap in enumerate(caps):
		with silence(True):
			H = deepcopy(G)
			trajs, stats = generate_traffic(H, file_traffic=file_traffic, capacity_factor=cap, put_sectors=True,
										remove_flights_after_midnight=True)
			# Get the fraction of flights rejected
			ratio = stats['rejected_flights']/float(stats['flights'])
		#print "ratio:", ratio
		# If the fraction is over the target, it is our value.
		if ratio>=target:
			break
		last_ratio = ratio

	# Second try with smaller step
	if i==len(caps)-1 and ratio<target:
		for i, cap in enumerate(caps):
			return find_good_scaling_capacity(G, file_traffic, silent=silent, max_value=caps[-1], min_value=min_value, step=step/10., target=target, **paras_control)

	# Refinement
	if ratio!=last_ratio and i!=0:
		cap = cap - (ratio - target)*(cap - caps[i-1])/(ratio - last_ratio)

	with silence(True):
		H = deepcopy(G)
		trajs, stats = generate_traffic(H, file_traffic=file_traffic, capacity_factor=cap, put_sectors=True,
											remove_flights_after_midnight=True)
				# Get the fraction of flights rejected
		ratio = stats['rejected_flights']/float(stats['flights'])

	return cap, ratio, H

if __name__=='__main__':
	airac = 334
	starting_date=[2010,5,6]
	n_days=1
	cut_alt=240.
	mode='navpoints'
	data_version=None
	n_iter = 100
	target_rejected_flights = 0.02
	
	#for country in ['LF', 'LE', 'EG', 'EB', 'LI']:
	for country in ['LI']:
		paras = paras_strategic(zone=country, airac=airac, starting_date=starting_date, n_days=n_days, cut_alt=cut_alt,\
			mode=mode, data_version=data_version)

		# db=_mysql.connect("localhost","root", paras['password_db'], "ElsaDB_A" + str(airac), conv=my_conv)

		# db.query("""SELECT accName FROM ACCBoundary WHERE accName LIKE '""" + country + """%'""")
		# r=db.store_result()
		# rr=[rrr[0] for rrr in r.fetch_row(maxrows=0,how=0)]
		# db.close()

		rr = ['LIRR']

		for zone in rr:
			print "=============================================="
			print "    Generating traffic for zone:", zone
			print "=============================================="
			paras_nav = paras_strategic(zone=zone, mode='navpoints', data_version=data_version) #TODO
			name_G = name_net(paras_nav, data_version)
			try:
				G = network_whose_name_is(_result_dir + '/networks/' + name_G)
			except IOError:
				print "Could not load the network, I skip it."
				print 
				continue

			with open('../libs/All_shapes_334.pic','r') as f:
				all_shapes = pickle.load(f)
			boundary = list(all_shapes[zone]['boundary'][0].exterior.coords)
			assert boundary[0]==boundary[-1]

			with open(_result_dir + '/trajectories/bounds/' + G.name + '_bound_latlon.dat', 'w') as f:
				for x, y in boundary:
					f.write(str(x) + '\t' + str(y) + '\n')

			print "Finding best capacity factor..."
			capacity_factor, rejected_flights, H = find_good_scaling_capacity(G, _result_dir + "/networks/" + name_G + '_flights_selected.pic', target=target_rejected_flights)
			print "Found best capacity factor:", capacity_factor, "(rejected fraction", rejected_flights, "of flights)"
			
			#print "Capacities of sectors:", {n:H.node[n]['capacity'] for n in H.nodes()}
			write_down_capacities(H, save_file=_result_dir + '/trajectories/capacities/' + G.name + '_capacities_rej' + str(target_rejected_flights) + '_new.dat')
			print "Capacities saved as", _result_dir + '/trajectories/capacities/' + G.name + '_capacities_rej' + str(target_rejected_flights) + '_new.dat' 
			for i in range(n_iter):
				counter(i, n_iter, message="Doing simulations...")
				name_results = name_sim(name_G) + '_rej' + str(target_rejected_flights) + '_new_' + str(i) + '.dat'
				with silence(True):
					trajs, stats = generate_traffic(deepcopy(G), save_file=_result_dir + '/trajectories/M1/' + name_results,
										record_stats_file=_result_dir + '/trajectories/M1/' + name_results.split('.dat')[0] + '_stats.dat',
										file_traffic=_result_dir + "/networks/" + name_G + '_flights_selected.pic',
										put_sectors=True,
										remove_flights_after_midnight=True,
										capacity_factor=capacity_factor)
				
			print 
			print 