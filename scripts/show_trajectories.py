#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This a a script which displays a set of trajectories loaded from the disk,
with the standard format of the Simulator (x, y, z, t, s)
"""

import sys
import os
sys.path.insert(1, '..')

from abm_strategic.utilities import read_trajectories_for_tact, draw_traffic_network

if __name__ == '__main__':
	traj_file = None if len(sys.argv)==1 else sys.argv[1]
	save_file = None if len(sys.argv)<2 else sys.argv[2]

	print 'You can specifiy a path to save the figure after the trajectory file.'
	if traj_file==None:
		raise Exception("Please give a trajectory file: ./show_trajectories file_trajectory.dat")

	print "Parsing the trajectories..."
	trajectories = read_trajectories_for_tact(traj_file)

	draw_traffic_network(trajectories, thr=0., already_in_degree=True,\
		sizes=15, save_file=save_file, show=False, weight_scale=0.25, alpha=0.7)


