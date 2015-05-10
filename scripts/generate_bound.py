#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
import os
import pickle
from os.path import join as jn

def write_down_area(area):
	with open('../libs/All_shapes_334.pic','r') as f:
		all_shapes = pickle.load(f)
		boundary = list(all_shapes[area]['boundary'][0].exterior.coords)
		assert boundary[0]==boundary[-1]

	with open('../abm_tactical/config/bound_latlon.dat', 'w') as f:
		for x, y in boundary:
			f.write(str(x) + '\t' + str(y) + '\n')

def write_down_sectors_from_network(G, rep=None):
	os.system('mkdir -p ' + rep)
	for n in G.nodes():
		boundary = list(G.polygons[n].exterior.coords)
		with open(jn(rep, str(n)+'_bound_latlon.dat'), 'w') as f:
			for x, y in boundary:
				f.write(str(x) + '\t' + str(y) + '\n')

if __name__ == '__main__':
	#write_down_area('LFMM')
	name_net = 'Real_LI_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed'
	with open('../../results/networks/' + name_net + '.pic') as _f:
		G = pickle.load(_f)
	write_down_sectors_from_network(G, rep='../../results/networks/' + name_net + '_sectors_boundaries')


