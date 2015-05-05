#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
========================================================
This file gathers several functions for plotting and 
analysis relevant results of the simulations and or of 
the network.
TODO: update, tests and documentation.
========================================================
"""

import sys
sys.path.insert(1,'..')
import matplotlib.pyplot as plt
from simulationO import build_path, Simulation
import pickle
import numpy as np
import os
import csv

from prepare_navpoint_network import extract_capacity_from_traffic, extract_old_capacity_from_traffic
#from utilities import network_and_trajectories_of_real_traffic, network_and_trajectories_of_simulated_traffic,\
# compare_networks as compare_networks_dist, read_paras

from libs.tools_airports import get_paras, extract_flows_from_data
from libs.general_tools import write_on_file, plot_scatter, plot_quantiles, plot_hist, logged

version = '2.9.1'


# def get_flights_from_flows(G):
# 	_paras=get_paras()
# 	_paras['zone']=G.country
# 	_paras['airac']=G.airac
# 	_paras['type_zone']='EXT'
# 	_paras['filtre']='Strong'
# 	_paras['mode']='navpoints'
# 	_paras['cut_alt']=240.
# 	_paras['both']=False
# 	_paras['n_days']=1
# 	_pairs = [(G.G_nav.node[_n1]['name'], G.G_nav.node[_n2]['name']) for _n1,_n2 in G.G_nav.short.keys()]
# 	flights = extract_flows_from_data(_paras, [G.G_nav.node[n]['name'] for n in G.G_nav.nodes()], pairs = _pairs)[2]

# 	return flights

"""
========================================================
"Static" functions to deal only with the network itself 
(not the result of the simulations)
========================================================
"""

@logged
def compare_capacities(paras, rep = '', save_to_csv = True):
	"""
	Compare new and old definition of capacities.
	"""	

	G = paras['G']
	paras_real = G.paras_real
	capacities_new = extract_capacity_from_traffic(G, paras_real)
	capacities_old = extract_old_capacity_from_traffic(G, paras_real)

	with open('/home/earendil/Documents/ELSA/Modules/All_areas_volumes_' + str(paras_real['airac']) + '.pic','r') as f:
		real_areas, volumes=pickle.load(f)

	sectors = sorted(capacities_new.keys())

	capacities_new, capacities_old, areas, volumes = [capacities_new[v] for v in sectors], [capacities_old[v] for v in sectors],\
	[real_areas[G.node[v]['name']] for v in sectors], [volumes[G.node[v]['name']] for v in sectors]

	if save_to_csv:
		for met, met_name in [(capacities_new, 'cap_new'), (capacities_old, 'cap_old'),\
				(areas, 'areas'), (volumes, 'vol')]:
			with open(rep + '/' + met_name + '_dist.csv', 'wb') as csvfile:
				writer = csv.writer(csvfile, delimiter=';')
				writer.writerow([met_name])
				for d in met:
					writer.writerow([d])

	os.system('mkdir -p ' + rep)

	plot_scatter(capacities_old, capacities_new, rep=rep, suffix='capacities', xlabel='Old capacities', ylabel='New capacities')
	plot_scatter(areas, capacities_new, rep=rep, suffix='area_new_capacities', xlabel='area', ylabel='New capacities')
	plot_quantiles(capacities_old, capacities_new, rep=rep, suffix='capacities', xlabel='Old capacities', ylabel='New capacities')
	plot_quantiles(areas, capacities_new, rep=rep, suffix='area_new_capacities', xlabel='area', ylabel='New capacities')

	plot_hist(capacities_new, rep=rep, suffix='capacities_new', xlabel='capacities_new')
	#plt.show()

@logged
def compute_basic_stats(paras, rep='', save_to_csv=False):
	"""
	Simple statistics.
	Beware! Here the weights are not given by the traffic. They are the time of crossing!
	"""
	G = paras['G'].G_nav

	os.system('mkdir -p ' + rep)

	print 'Number of nodes, number of edges:', len(G.nodes()), len(G.edges())

	# ----- Distribution of degree ----- #
	deg = [G.degree(n) for n in G.nodes()]
	bins = max(deg) - min(deg)
	if save_to_csv:
		with open(rep + '/deg_dist.csv', 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=';')
			writer.writerow(['Degree'])
			for d in deg:
				writer.writerow([d])

	print ''
	print 'Min/Mean/Std/Max degree:', min(deg), np.mean(deg), np.std(deg), max(deg)

	plot_hist(deg, xlabel = 'Degree', title = 'Distribution of degree', bins = bins, rep= rep, suffix = 'deg')

	# ----- Distribution of strength ----- #
	strr = [G.degree(n, weight = 'weight') for n in G.nodes()]
	bins = max(strr) - min(strr)
	if save_to_csv:
		with open(rep + '/str_dist.csv', 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=';')
			writer.writerow(['Strength'])
			for s in strr:
				writer.writerow([s])

	print ''
	print 'Min/Mean/Std/Max stregth:', min(strr), np.mean(strr), np.std(strr), max(strr)

	plot_hist(strr, xlabel = 'Strength', title = 'Distribution of strength', bins = bins, rep= rep, suffix = 'str')


	# ----- Distribution of weights ----- #
	wei = [G[e[0]][e[1]]['weight'] for e in G.edges()]
	bins = 30

	if save_to_csv:
		with open(rep + '/wei_dist.csv', 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=';')
			writer.writerow(['Weights'])
			for w in wei:
				writer.writerow([w])

	print ''
	print 'Min/Mean/Std/Max weight:', min(wei), np.mean(wei), np.std(wei), max(wei)
	print 'Total weight:', sum(wei)

	plot_hist(wei, xlabel='Time of travel between navpoints', title='', bins=bins, rep=rep, suffix='wei')


"""
========================================================
"Dynamic" functions to deal only with the results of the
simulations
========================================================
"""

@logged
def compare_networks(paras, rep=''):
	"""
	Comparing real network with simulated one.
	"""
	with open(build_path(paras) + '/sim.pic', 'r') as f:
		sim = pickle.load(f)

	os.system('mkdir -p ' + rep)

	G_real, traj_real = network_and_trajectories_of_real_traffic(sim)
	G_sim, traj_sim = network_and_trajectories_of_simulated_traffic(sim)

	# Number of nodes, number of edges.

	print 'Real: number of nodes and edges:', len(G_real.nodes()), len(G_real.edges())
	print 'Simu: number of nodes and edges:', len(G_sim.nodes()), len(G_sim.edges())

	# ----- Distribution of degree ----- #
	deg_real = [G_real.degree(n) for n in G_real.nodes()]
	deg_sim = [G_sim.degree(n) for n in G_sim.nodes()]
	bins_real = max(deg_real) - min(deg_real)
	bins_sim = max(deg_sim) - min(deg_sim)
	bins = max(bins_real, bins_sim)

	print  ''
	print 'Real: Min/Mean/Std/Max degree:', min(deg_real), np.mean(deg_real), np.std(deg_real), max(deg_real)
	print 'Simu: Min/Mean/Std/Max degree:', min(deg_sim), np.mean(deg_sim), np.std(deg_sim), max(deg_sim)
	print ''

	plt.figure()
	plt.title('Distribution of degree')
	# plt.hist(deg_real, bins_real, facecolor = 'red', alpha = 0.3, label = 'real')
	# plt.hist(deg_sim, bins_sim, facecolor = 'green', alpha = 0.3, label = 'sim.')
	plt.hist([deg_sim, deg_real], bins, color = ['orange', 'blue'], alpha = 0.6, label = ['sim.', 'real'])
	plt.ylabel('Counts')
	plt.xlabel('Degree')
	plt.legend()
	plt.savefig(rep + '/hist_deg.png')

	# ----- Distribution of strength ----- #
	str_real = [G_real.degree(n, weight = 'weight') for n in G_real.nodes()]
	str_sim = [G_sim.degree(n, weight = 'weight') for n in G_sim.nodes()]
	bins_real = 20
	bins_sim = 20

	print 'Real: Min/Mean/Std/Max strength:', min(str_real), np.mean(str_real), np.std(str_real), max(str_real)
	print 'Simu: Min/Mean/Std/Max strength:', min(str_sim), np.mean(str_sim), np.std(str_sim), max(str_sim)
	print ''

	plt.figure()
	plt.title('Distribution of strength')
	# plt.hist(str_real, bins_real, facecolor = 'red', alpha = 0.3, label = 'real')
	# plt.hist(str_sim, bins_sim, facecolor = 'green', alpha = 0.3, label = 'sim.')
	plt.hist([str_sim, str_real], bins_sim, color = ['orange', 'blue'], alpha = 0.6, label = ['sim.', 'real'])
	plt.ylabel('Counts')
	plt.xlabel('Strength')
	plt.legend()
	plt.savefig(rep + '/hist_str.png')

	# ----- Distribution of weights ----- #
	wei_real = [G_real[e[0]][e[1]]['weight'] for e in G_real.edges()]
	wei_sim = [G_sim[e[0]][e[1]]['weight'] for e in G_sim.edges()]
	bins_real = 20
	bins_sim = 20

	print 'Real: Min/Mean/Std/Max weight:', min(wei_real), np.mean(wei_real), np.std(wei_real), max(wei_real)
	print 'Real: total weight:', sum(wei_real)
	print 'Simu: Min/Mean/Std/Max weight:', min(wei_sim), np.mean(wei_sim), np.std(wei_sim), max(wei_sim)
	print 'Simu: total weight:', sum(wei_sim)
	print ''

	plt.figure()
	plt.title('Distribution of weight')
	# plt.hist(wei_real, bins_real, facecolor = 'red', alpha = 0.3, label = 'real')
	# plt.hist(wei_sim, bins_sim, facecolor = 'green', alpha = 0.3, label = 'sim.')
	plt.hist([wei_sim, wei_real], bins_sim, color = ['orange', 'blue'], alpha = 0.6, label = ['sim.', 'real'])
	plt.ylabel('Counts')
	plt.xlabel('Weights')
	plt.legend()
	plt.savefig(rep + '/hist_wei.png')


	# ------ Distribution of length of trajectories (without weight) ------- #

	len_real = [len(t) for t in traj_real]
	len_sim = [len(t) for t in traj_sim]
	bins_real = 20
	bins_sim = 20

	print 'Real: Min/Mean/Std/Max length of trajectories (without weight):', min(len_real), np.mean(len_real), np.std(len_real), max(len_real)
	print 'Simu: Min/Mean/Std/Max length of trajectories (without weight):', min(len_sim), np.mean(len_sim), np.std(len_sim), max(len_sim)
	print ''

	plt.figure()
	plt.title('Distribution of topological length')
	#plt.hist(len_real, bins_real, facecolor = 'red', alpha = 0.6, label = 'real')
	#plt.hist(len_sim, bins_sim, facecolor = 'green', alpha = 0.6, label = 'sim.')
	plt.hist([len_sim, len_real], bins_sim, color = ['orange', 'blue'], alpha = 0.6, label = ['sim.', 'real'])
	plt.ylabel('Counts')
	plt.xlabel('Length')
	plt.legend()
	plt.savefig(rep + '/hist_len.png')


	# ------ Distribution of length of trajectories ------- #

	# len_wei_real = [sim.G.G_nav.weight_path(t) for t in traj_real]
	# len_wei_sim = [sim.G.G_nav.weight_path(t) for t in traj_sim]
	# bins_real = 20
	# bins_sim = 20

	# print 'Real: Min/Mean/Std/Max length of trajectories (without weight):', min(len_wei_real), np.mean(len_wei_real), np.std(len_wei_real), max(len_wei_real)
	# print 'Simu: Min/Mean/Std/Max len_weigth of trajectories (without weight):', min(len_wei_sim), np.mean(len_wei_sim), np.std(len_wei_sim), max(len_wei_sim)
	# print ''

	# plt.figure()
	# plt.title('Distribution of length (weighted)')
	# plt.hist(len_wei_real, bins_real, facecolor = 'yellow', alpha = 0.3, label = 'real')
	# plt.hist(len_wei_sim, bins_sim, facecolor = 'blue', alpha = 0.3, label = 'sim.')
	# plt.legend()
	# plt.savefig(rep + '/hist_len_wei.png')

	#plt.show()

if __name__=='__main__':
	#compare_networks(paras, rep = build_path(paras) + '/analysis')
	compare_capacities(read_paras(), rep=paras['G'].name+'/compare_capacities')
	#compute_basic_stats(paras, rep= paras['G'].name + '/basic_stats', save_to_csv = True)

	#plt.show()
	

	



