#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../..')
sys.path.insert(1, '../../abm_strategic')
import pickle
import os 
from os.path import join as jn

from abm_strategic.simulationO import generate_traffic
from abm_strategic import result_dir, main_dir
#result_dir = jn(main_dir, 'tests', 'example')

if __name__ == '__main__':
	with open(jn(result_dir, 'networks', 'Example.pic'), 'r') as f:
		G = pickle.load(f) 

	os.system('mkdir -p ' + jn(result_dir, 'trajectories', 'M1'))
	save_file = jn(result_dir, 'trajectories', 'M1', 'trajectories_example.dat')
	paras_file = jn(main_dir, 'abm_strategic', 'my_paras.py')

	trajectories = generate_traffic(G, 	paras_file=paras_file,
										save_file=save_file, 
										# file_traffic=file_traffic, 
										# coordinates=True,
										# put_sectors=True, 
										# save_file_capacities=save_file_capacities,
										# capacity_factor=0.05,
										# remove_flights_after_midnight=True,
										# record_stats_file='../trajectories/M1/trajectories_stats.dat'
										# rectificate={'eff_target':0.99, 'inplace':False, 'hard_fixed':False, 'remove_nodes':True, 'resample_trajectories':True},
										# storymode=False,
										# ACtot=4000
										)