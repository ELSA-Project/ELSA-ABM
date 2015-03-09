#!/usr/bin/env python
"""
Generation of altitudes for flights given a distribution of flight levels.

@author: earendil
"""
import sys
sys.path.insert(1,'../abm_strategic')
sys.path.insert(1,'..')
import os

from scipy import stats
from string import split, strip

from simAirSpaceO import FlightPlan
from general_tools import getDistribution
from utilities import write_trajectories_for_tact
from paths import result_dir

def load_from_trajectories(fil='../trajectories/trajectories.dat', starting_date = [2010, 6, 5, 10, 0, 0]):
	"""    
	Read a file in the abm_tactical format
	"""
	trajectories = []
	with open(fil, 'r') as f:
		content = f.readlines()

	for line in content[1:]:
		stuff = split(strip(line), "\t")
		traj = split(stuff[2], ",")
		x = traj[0]
		y = traj[1]
		z = traj[2]

	# for trajectory in enumerate(trajectories):
	#     print >>f, str(i) + "\t" + str(len(trajectory)) + '\t',
	#     for x, y, z, t in trajectory:
	#         print >>f, str(x) + "," + str(y) + "," + str(z) + "," + str(t) + '\t',
	#     print >>f, ''

	print "Trajectories saved in", fil  

def generate_altitudes_for_traj(trajectories, distr_file = None, distr_type = "flat", min_FL = 240., max_FL = 350., save_file = None, starting_date = [2010, 6, 5, 10, 0, 0]):
	"""
	@trajectories: a list of tuple (lat, lon, alt, time).
	TODO: do a distribution for entry and for exit?
	"""
	print "Generating altitudes from distribution..."
	trajectories = [[list(p) for p in traj] for traj in trajectories]

	if distr_file!=None:
		print "Getting distribution of altitudes from file", distr_file
		distr_type = "data"
		data = []
		with open(distr_type, 'r') as f:
			for columns in (raw.strip().split() for raw in f):  
				data.append(columns[0])
		min_FL, max_FL = min(data), max(data)
		distr = getDistribution(data)
	else:
		if distr_type == 'flat':
			distr = stats.randint(low = min_FL, high = max_FL).rvs
		else:
			print "You asked for a distribution of type", distr_type
			raise Exception("This type of distribution is not implemented.")

	for traj in trajectories:
		alt = distr()
		for p in traj: #same altitude for the whole trajectory
			p[2] = 10*int(alt/10.) # To have trajectories separated by 10 FL.



	if save_file!=None:
		write_trajectories_for_tact(trajectories, fil = save_file, starting_date = starting_date)

	return trajectories

if __name__ == '__main__':
	from simulationO import generate_traffic
	import pickle

	with open("../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2.pic", "r") as f:
		G = pickle.load(f)		
	trajectories = generate_traffic(G, #save_file = '../trajectories/trajectories.dat', ACtot = 10)
					ACtot = 1000)

	trajectories = generate_altitudes_for_traj(trajectories, save_file = os.path.join(result_dir, 'trajectories/M1/trajectories_alt.dat'))


	# print
	# print
	# print "Trajectories:"
	# for traj in trajectories:
	# 	print traj 
	# 	print


