#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1, '..')

import os
from string import split
import numpy as np
import pickle

from abm_tactical import *
from abm_strategic import *
from abm_strategic.utilities import draw_network_map

def find_group(element, groups):
	for g, els in groups.items():
		for el in els:
			if el == element: break

	return g
	#b = {c:g for g in a.keys() for c in a[g]}


def do_plop():
	#from abm_strategic.paras_G import paras_G
	#G = prepare_hybrid_network(paras_G)


	sys.path.insert(1, '../abm_strategic')
	with open("../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2.pic", 'r') as f:
		G = pickle.load(f)

	def get_coords(nvp):
		return G.G_nav.node[nvp]['coord']

	def add_node(trajs, G, coords, f, p, groups, group_new_node, dict_nodes_traj):
		if len(dict_nodes_traj[trajs[f][p]])>1:
			dict_nodes_traj[trajs[f][p]].remove(f)
		else:
			del dict_nodes_traj[trajs[f][p]]
			g = find_group(trajs[f][p], groups)
			groups[g].remove(trajs[f][p])

		new_node = len(G.nodes())
		G.add_node(new_node, coord = coords)
		trajs[f][p] = new_node
		groups[group_new_node].append(new_node)
		dict_nodes_traj[new_node] = [f]

		return trajs, G, groups, dict_nodes_traj

	
	trajectories = generate_traffic(G, paras_file = '../abm_strategic/paras.py', save_file = None,  simple_setup=True, 
		starting_date = [2010, 6, 5, 10, 0, 0], coordinates = False, ACtot=3)
	#def d((p1, p2)):
	#	return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

	def d((n1, n2)):
		p1 = get_coords(n1)
		p2 = get_coords(n2)
		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

	print trajectories
	print
	probabilities = {n:1./float(len(G.G_nav.nodes())) for n in G.G_nav.nodes()}	
	#print probabilities.keys()
	traj_eff = rectificate_trajectories(trajectories, 0.995, dist_func = d, add_node_func = add_node, coords_func =  get_coords,
		 G = G.G_nav, probabilities = probabilities)

	print traj_eff

	#draw_network_map(G, title='Network map', trajectories=[], rep='./',airports=True, load=True, generated=False, add_to_title='', polygons=[], numbers=False, show=True):
    


def do_ABM_tactical(input_file, output_file):
	inpt = ["", input_file, output_file]

	print "M1 source:", inpt[1]
	print "Destination output:", inpt[2]
	print
	print
	print "Running ABM Tactical model..."

	tactical_simulation(inpt)

	print
	print
	print "Done."


if __name__ == '__main__':
	main_dir = os.path.abspath(__file__)
	main_dir = os.path.split(os.path.dirname(main_dir))[0]
	print main_dir

	print
	#input_file = os.path.join(main_dir, "trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat")
	input_file = os.path.join(main_dir, "trajectories/M1/trajectories_alt.dat")
	output_file = os.path.join(main_dir, "results/output.dat")
	#do_ABM_tactical(input_file, output_file)
	do_plop()


