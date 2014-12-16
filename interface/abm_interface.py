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
from abm_strategic.utilities import draw_network_map, select_interesting_navpoints, OD


if 1:
    # Manual seed
    see_=1
    #see_ = 15 #==> the one I used for new, old_good, etc.. 
    # see=2
    print "===================================="
    print "          USING SEED", see_
    print "===================================="
    np.random.seed(see_)


def do_plop():
	#from abm_strategic.paras_G import paras_G
	#G = prepare_hybrid_network(paras_G)

	# Choice of the network
	sys.path.insert(1, '../abm_strategic')
	with open("../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2.pic", 'r') as f:
		G = pickle.load(f)

	# Choice of the trajectories
	trajectories = generate_traffic(G, paras_file='../abm_strategic/paras.py', save_file=None, simple_setup=True, 
		starting_date=[2010, 6, 5, 10, 0, 0], coordinates=False, ACtot=100)

	# Make groups
	n_best = select_interesting_navpoints(G, OD = OD(trajectories), N_per_sector = 1) # Selecting points with highest betweenness centrality within each sector
	n_best = [n for sec, points in n_best.items() for n in points]
	# groups = {}
	# for n in G.G_nav.nodes():
	# 	gg = np.random.choice(["A", "B"])
	# 	groups[gg] = groups.get(gg, []) + [n]
	# probabilities = {"A":0.7, "B":0.3}

	groups = {"C":[], "N":[]} # C for "critical", N for "normal"
	for n in G.G_nav.nodes():
		if n in n_best:
			groups["C"].append(n)
		else:
			groups["N"].append(n)
	probabilities = {"C":0.001, "N":0.999}

	draw_network_map(G.G_nav, title='Network map', trajectories=trajectories, rep='./', airports=False, load=False, generated=True, add_to_title='', polygons=G.polygons.values(), numbers=False, show=False)

	traj_eff = rectificate_trajectories_network(trajectories, 0.995, G, groups=groups, probabilities=probabilities, remove_nodes = True)

	draw_network_map(G.G_nav, title='Network map', trajectories=traj_eff, rep='./', airports=False, load=False, generated=True, add_to_title='', polygons=G.polygons.values(), numbers=False, show=True)


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


