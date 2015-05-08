# -*- coding: utf-8 -*-

"""
Gathers utilities used for studying and changing the efficiency of trajectories.
The effciency is the ratio between the length of the trajectory and the corresponding
ideal trajectory, i.e. the staight line (or grand circle).
"""

import sys
sys.path.insert(1, '..')

from collections import Counter 
import pylab as pl
import random as rd
import numpy as np
from numpy.random import choice, seed
import math as mt
from shapely.geometry import LineString, Point
from copy import deepcopy
from datetime import datetime, timedelta

#from igraph import *

from libs.tools_airports import build_long_2d
from libs.general_tools import insert_list_in_list

__version__ = "1.4"

# Several handy functions.
gall = lambda x: [6371000.*pl.pi*x[1]/(180.*pl.sqrt(2)), 6371000.*pl.sqrt(2)*pl.sin(pl.pi*(x[0]/180.))]
invgall = lambda x: [(180./pl.pi)*pl.arcsin(x[1]/(6371000.*pl.sqrt(2))) , x[0]*180*pl.sqrt(2)/(6371000.*pl.pi)]
to_str = lambda x: str(x[0][0])+' '+str(x[0][1])
dist = lambda x: pl.sqrt( (x[0][0]-x[1][0])**2 + (x[0][1]-x[1][1])**2 )
to_coord = lambda z: gall([float(z.split(' ')[0]),float(z.split(' ')[1])])
to_coord_ng = lambda z : [float(z.split(' ')[0]),float(z.split(' ')[1])]
pday = lambda x:  datetime.strftime(x,"%Y-%m-%d")
to_OD = lambda z: to_str([z[0]])+','+to_str([z[-1]])
to_OD2 = lambda z: to_str([z[0]])+','+to_str([z[1]])
to_OD_inv  = lambda z: to_str([invgall(z[0])])+','+to_str([invgall(z[-1])])
tf_time= lambda z: datetime.strptime("2010-06-02 0:0:0:0",'%Y-%m-%d %H:%M:%S:%f') +timedelta(seconds=((int((z - datetime.strptime("2010-06-02 0:0:0:0",'%Y-%m-%d %H:%M:%S:%f') ).total_seconds()) /4)*4))


def compute_efficiency(trajectories, dist_func=dist):
	"""
	Compute the efficiency of a set of trajectories. In general, the efficiency is the ratio
	between the length of the trajectories and the ideal corresponding trajectory, i.e. the 
	straight line or the grand circle distance. Here the function computes the ratio between 
	the sum of the length of all trajectories and the sum of the ideal length.

	Parameters
	----------
	trajectories : list
		of trajectories with signature (x, y)
	dist_func : function, optional
		used to compute the distance between two points. by default it is the euclidean distance.

	Returns
	-------
	float, float : the efficiency and the sum of the lengths ideal trajectories.

	"""

	L = [sum([dist_func([trajectories[f][p-1],trajectories[f][p]]) for p in range(1, len(trajectories[f]))]) for f in range(len(trajectories))]
	S = [dist_func([trajectories[f][0],trajectories[f][-1]]) for f in range(len(trajectories))]

	return sum(S)/sum(L), sum(S)

def find_group(element, groups):
	"""
	Find which group element belongs to. If element belongs to several groups,
	returns only one of them (randomly if groups is not an sorted dict).

	Parameters
	----------
	element : object
	group : dictionnary
		where keys are labels of groups and values are list of objects.

	Returns
	-------
	group : label
		group whose element belongs to.

	Notes
	-----
	Not used at the moment.
	Beware: use b = {c:g for g in a.keys() for c in a[g]}

	"""

	for g, els in groups.items():
		for el in els:
			if el == element:
				return g

def find_sector(navpoint, G):
	"""
	Finds to which sector belongs the navpoint based on geometrical boundaries of navpoints.

	Parameters
	----------
	navpoint : label
	G : Hybrid network
		must have the sectors stored in the 'shapes' attribute as Shapely Polygons.

	Returns
	-------
	sector : label
		Returns None if the navpoint does no belong to any sector.

	"""

	for sec, pol in G.polygons.items():
		if pol.contains(Point(G.G_nav.node[navpoint]['coord'])):
			return sec

def return_linear_interpolation_altitude_func(long_dis_cul, alt):
	"""
	Build an simple linear interpolation function for the altitudes. 

	Parameters
	----------
	long_dis_cul : list of float
		list of linear distances between each navpoint of a trajectory.
	alt : list of float
		list of altitudes reached at each navpoint.

	Returns
	-------
	f : function 
		to a float x representing a linear distance from the beginning of
		a trajectory, returns the interpolated altitude at this point. 
	
	"""

	def f(x):
		if x>0.:
			coin = list(long_dis_cul<x) 
			if False in coin:
				idx_aft = coin.index(False)
				idx_bef = idx_aft - 1
				return int(alt[idx_bef] + (alt[idx_aft] - alt[idx_bef])*(x - long_dis_cul[idx_bef])/(long_dis_cul[idx_aft] - long_dis_cul[idx_bef]))
			else:
				return alt[-1]
		else:
			return alt[0]
	return f

def rectificate_trajectories_network_with_time_and_alt(trajs_w_t, eff_target, G, remove_nodes=False, 
	resample_trajectories=True, **kwargs_rectificate):
	"""
	Same than rectificate_trajectories_network_with_time, but recomputes altitudes with linear interpolation.

	Parameters
	----------
	trajs_w_t : list 
		of trajectories with signatures (n, z), t
	eff_target : float
		target efficiency.
	G : networkx Graph or DiGraph.
		Navpoint network.
	remove_nodes : boolean, optional
		if True, remove the chosen nodes from the trajectories. Otherwise, duplicate and move the existing point.
	resample_trajectories : boolean, optional
		If True and remove_nodes is True, the function will recreate navpoint on the trajectories after the 
		rectification procedure so as to have the same number of navpoints than in the original trajectory. 
		The napvoints are equally spaced along the trajectory.
	kwargs_rectificate : additional parameters passed to rectificate_trajectories.

	Returns
	--------
	trajs : list 
		of modified trajectories with signature (n, z), t
	eff : float
		corresponding efficiency.
	G : networkx graph
		with added or removed nodes.
	groups : dictionnary
		the final groups.

	"""

	# Compute geometrical trajectories (with altitudes)
	geom_trajs_alt, start_dates = zip(*trajs_w_t)
	#geom_trajs_alt = list(geom_trajs)
	geom_trajs, alts = [], []
	for traj in geom_trajs_alt:
		g_t, alt = zip(*traj)
		geom_trajs.append(list(g_t))
		alts.append(list(alt))

	# Rectificate geometrical trajectories
	G_old = deepcopy(G)
	trajs_rec, eff, G, groups_rec = rectificate_trajectories_network(geom_trajs, eff_target, G, remove_nodes=remove_nodes, resample_trajectories=resample_trajectories, **kwargs_rectificate)

	# Recompute altitudes
	for i, traj_new in enumerate(trajs_rec):
		traj_old = geom_trajs[i]
		long_dis, long_dis_cul_old = build_long_2d([G_old.node[n]['coord'] for n in traj_old]) 
		long_dis_cul_old = np.array(long_dis_cul_old)
		alt = alts[i]

		long_dis, long_dis_cul = build_long_2d([G.node[n]['coord'] for n in traj_new]) 
		long_dis_cul = np.array(long_dis_cul)
		
		f_inter = return_linear_interpolation_altitude_func(long_dis_cul_old, alt)

		new_alt = [f_inter(d*long_dis_cul_old[-1]/long_dis_cul[-1]) for d in long_dis_cul]

		trajs_rec[i] = list(zip(traj_new, new_alt))

	# Put back the starting date.
	trajs_rec_w_t = list(zip(trajs_rec, start_dates))

	return trajs_rec_w_t, eff, G, groups_rec

def rectificate_trajectories_network_with_time(trajs_w_t, eff_target, G, remove_nodes=False, resample_trajectories=True, **kwargs_rectificate):
	"""
	Same than rectificate_trajectories_network for trajectories including a starting date.

	Parameters
	----------
	trajs_w_t : list 
		of trajectories with signatures (n), tt or (n), t
	eff_target : float
		target efficiency.
	G : networkx Graph or DiGraph.
		navpoint network.
	remove_nodes : boolean, optional
		if True, remove the chosen nodes from the trajectories. Otherwise, duplicate and move the existing point.
	resample_trajectories : boolean, optional
		If True and remove_nodes is True, the function will recreate navpoint on the trajectories after the 
		rectification procedure so as to have the same number of navpoints than in the original trajectory. 
		The napvoints are equally spaced along the trajectory.
	kwargs_rectificate : additional parameters passed to rectificate_trajectories.

	Returns
	--------
	trajs : list 
		of modified trajectories with signature (n), tt or (n), t
	eff : float
		corresponding efficiency.
	G : networkx graph
		with added or removed nodes.
	groups : dictionnary
		the final groups.

	"""

	# Compute geometrical trajectories
	geom_trajs, start_dates = zip(*trajs_w_t)
	geom_trajs = list(geom_trajs)

	# Rectificate geometrical trajectories
	trajs_rec, eff, G, groups_rec = rectificate_trajectories_network(geom_trajs, eff_target, G, remove_nodes=remove_nodes, resample_trajectories=resample_trajectories, **kwargs_rectificate)

	# Put back the starting date.
	trajs_rec_w_t = list(zip(trajs_rec, start_dates))

	return trajs_rec_w_t, eff, G, groups_rec

def rectificate_trajectories_network(trajs, eff_target,	G, remove_nodes=False, resample_trajectories=True, **kwargs_rectificate):
	"""
	Wrapper of rectificate_trajectories to use efficiently with a networkx object. 
	Provide default functions to add nodes to network. Resample also the trajectories in the case
	where the 'remove_nodes' mode is chosen. Propagates the kwargs for the rectification.
	Signature of each trajectory : (n).

	Parameters
	----------
	trajs : list 
		of trajectories with signatures (n).
	eff_target : float
		target efficiency.
	G : hybrid network
		The label of the navpoints must be integers.
	remove_nodes : boolean, optional
		if True, remove the chosen nodes from the trajectories. Otherwise, duplicate and move the existing point.
	resample_trajectories : boolean, optional
		If True and remove_nodes is True, the function will recreate navpoint on the trajectories after the 
		rectification procedure so as to have the same number of navpoints than in the original trajectory. 
		The napvoints are equally spaced along the trajectory.
	kwargs_rectificate : additional parameters passed to rectificate_trajectories.
	
	Returns
	--------
	trajs : list 
		of modified trajectories.
	eff : float
		corresponding efficiency.
	G : networkx graph
		with added or removed nodes.
	groups : dictionnary
		the final groups.
	
	Notes
	-----
	TODO: test for DiGraph.
	Remark: The nodes are never removed from the network, only from the trajectories.
	Changed in 1.4: fixed sector allocations of new nodes. Using an hybrid netwrok instead of a navpoint network.

	"""

	assert not -10000 in [G.G_nav.node[n]['sec'] for n in G.G_nav.nodes()]

	# Function to get the coordinates of the nodes on the network.
	def get_coords(nvp):
		try:
			pouet = G.G_nav.node[nvp]['coord']
		except:
			print
			print nvp
			print G.G_nav.node[nvp]
			raise
		return G.G_nav.node[nvp]['coord']
		
	def d((n1, n2)):
		p1 = get_coords(n1)
		p2 = get_coords(n2)
		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

	# Function to get add a node to the network.
	def add_node(trajs, G, coords, f, p):
		# G here is the navpoint network.
		new_node = max(G.nodes())+1
		#G.add_node(new_node, coord=coords, sec=G.node[trajs[f][p-1]]['sec'])
		G.add_node(new_node, coord=coords, sec=-10000)

		weight = G[trajs[f][p-1]][trajs[f][p]]['weight'] * d((trajs[f][p-1], new_node))/d((trajs[f][p-1], trajs[f][p]))
		G.add_edge(trajs[f][p-1], new_node, weight=weight)
		weight = G[trajs[f][p]][trajs[f][p+1]]['weight'] * d((trajs[f][p+1], new_node))/d((trajs[f][p], trajs[f][p+1]))
		G.add_edge(new_node, trajs[f][p+1], weight=weight)
		trajs[f][p] = new_node

		return new_node, trajs, G

	old_max_label = max(G.G_nav.nodes())
	trajs_old = deepcopy(trajs)
	trajs_rec, eff, G.G_nav, groups_rec = rectificate_trajectories(trajs, eff_target, 
																add_node_func=add_node, 
																dist_func=d, 
																coords_func=get_coords,
																G=G.G_nav, 
																#inplace=True,
																remove_nodes=remove_nodes, 
																**kwargs_rectificate)

	if remove_nodes and resample_trajectories: # Resample trajectories.
		for i, traj in enumerate(trajs_rec):
			n_old = len(trajs_old[i])
			n_new = len(traj)
			n_gen = n_old - n_new

			# Compute linear distance after each point.
			long_dis, long_dis_cul = build_long_2d([G.G_nav.node[n]['coord'] for n in traj]) 
			long_dis, long_dis_cul = np.array(long_dis), np.array(long_dis_cul)

			# Equally spaced points along the trajectory.
			long_dis_new_points = [(j+1.)*(long_dis_cul[-1]-long_dis_cul[0])/float(n_gen+1.) for j in range(n_gen)] 
			
			new_point_coords = []
			new_point_indices = []
			for dd in long_dis_new_points:
				point_before = len(long_dis_cul[dd>long_dis_cul])-1#.index(True) # Index of point before future point

				dn = np.array(get_coords(traj[point_before+1])) - np.array(get_coords(traj[point_before])) # direction vector
				dn = dn/np.sqrt(sum(dn**2))#np.norm(dn)
				new_point_coords.append(list(np.array(get_coords(traj[point_before])) + (dd-long_dis_cul[point_before])*dn))
				new_point_indices.append(point_before)

			# Names (indices) of new navpoint in the network
			names = [max(G.G_nav.nodes()) + 1 + j for j in range(len(new_point_indices))]

			traj_rec_without_new_points = deepcopy(trajs_rec[i])
			trajs_rec[i] = insert_list_in_list(trajs_rec[i], names, new_point_indices)

			# First add all new nodes to network
			for j, name in enumerate(names):
				# TODO: do this more carefully.
				# if 'sec' in G.G_nav.node[traj_rec_without_new_points[new_point_indices[j]]].keys():
				#	sec = G.G_nav.node[traj_rec_without_new_points[new_point_indices[j]]]['sec']
				#else:
				#	sec = 0	
				G.G_nav.add_node(name, coord=new_point_coords[j], sec=-10000)

			# Then add all edges
			for j in range(len(trajs_rec[i])-1):
				if not trajs_rec[i][j+1] in G.G_nav.neighbors(trajs_rec[i][j]):
					# TODO: improve this?
					ref_point_bef = trajs_old[i][0]
					ref_point_aft = trajs_old[i][1]
					scale_weight = G.G_nav[ref_point_bef][ref_point_aft]['weight']/d((ref_point_bef, ref_point_aft))

					name_pt_bef = trajs_rec[i][j]


					weight = scale_weight * d((trajs_rec[i][j], trajs_rec[i][j+1]))
					G.G_nav.add_edge(trajs_rec[i][j], trajs_rec[i][j+1], weight=weight)

			# Check that all consecutive points of the trajectory are neighbors in the network
			for j in range(len(trajs_rec[i])-1):
				try:
					assert trajs_rec[i][j+1] in G.G_nav.neighbors(trajs_rec[i][j])
				except AssertionError:
					print trajs_rec[i][j], trajs_rec[i][j+1]
					raise

	# Find the sector containing the new nodes
	for n in G.G_nav.nodes():
		if n>old_max_label:
			sec = find_sector(n, G)
			G.G_nav.node[n]['sec'] = sec
			G.node[sec]['navs'].append(n)

	return trajs_rec, eff, G, groups_rec

def rectificate_trajectories(trajs, eff_target, G=None, groups={}, add_node_func=None, dist_func=dist, coords_func=lambda x: x, \
	n_iter_max=1000000, probabilities={}, remove_nodes=False, hard_fixed=False, inplace=False):
	"""
	Given all trajectories and a value of efficiency, rectify the trajectories. The rectification can be 
	done in two ways: either some intermediate navpoints are removed from the trajectories,	or they are 
	moved to another position which straightens up the segment. The points are gathered in groups defined by the 
	user with kwarg 'groups'. These groups have different probabilities of being chosen. Then a navpoint is 
	chosen in this group, and finally a flight which crosses the navpoint is chosen. This goes on until the 
	target efficiency is met or or the stop criteria are met.

	Parameters
	-----------
	trajs : list of trajectories
		list of nodes (n) of network G if G is specified, or list of coordinates (x,y).
	eff_target : float
		target for efficiency.
	add_node_func : function, optional 
		encodes how to add the new nodes. 
		Signature: trajectories, network, coordinates of the new point, index of flight, index of p in trajectory of flight
	dist_func : function, optional
		associates a distance to a couple of points.
		Signature: 2-tuple of points from trajectories.
	coords_func : function, optional
		associates some coordinates to a point in a trajectory.
		Signature: point from a trajectory.
	n_iter_max : int, optional
		maximum number of iteration to reach target efficiency.
	G : networkx object, optional
		which could contain coordinates of points. If None, all trajectories should be coordinates-based.
	groups : dictionnary, optional
		Groups of nodes which don't have the same probability of being rectified. Keys are labels of groups, values 
		are lists of the nodes.
	probabilities : dictionnary, optional
		with the labels of groups as keys and the probabilities of all points of being rectified as values.
	remove_nodes : boolean, optional
		if True, remove the chosen nodes from the trajectories. Otherwise, duplicate and move the existing point.
	hard_fixed : boolean, optional
		if False, groups having 0 probability of being chosen will switch to non-zero probablity once every other 
		groups are empty. If True, the function exits when all groups are empty or with zero probability.
	inplace : boolean, optional
		if True, modifies trajs, G and groups. Otherwise return a copy of the objects.
 
	Returns
	--------
	trajs : list 
		of modified trajectories.
	eff : float
		corresponding efficiency.
	G : networkx graph
		with added or removed nodes.
	groups : dictionnary
		the final groups.

	Notes
	-----
	Changed in 1.1: efficiency is computed internally. Stop criteria on efficiency is < instead of <=. Added kwarg dist_func.
	Changed in 1.2: added dist_func, coords_func, n_iter_max, add_node_func, G.
	Changed in 1.3: added probabilities. Broken the legacy with coordinates-based trajectories.
	Added groups, remove_nodes, hard_fixed.

	"""

	if not inplace:
		trajs_rec = deepcopy(trajs)
	else:
		trajs_rec = trajs

	def add_node_func_coords(trajs, G, coords, f, p):
		trajs_rec[f][p][0] = coords[0]
		trajs_rec[f][p][1] = coords[1]

		return coords, trajs_rec, G

	if add_node_func==None: 
		add_node_func = add_node_func_coords

	print "Rectificating trajectories..."
	eff, S = compute_efficiency(trajs_rec, dist_func=dist_func)
	print "Initial efficiency/target efficiency:", eff, '/', eff_target
	Nf = len(trajs_rec)

	# To each point, associate the list of flights going through.
	dict_nodes_traj = {}
	for f in range(Nf):
		for p in trajs_rec[f][1:-1]:
			dict_nodes_traj[p] = dict_nodes_traj.get(p, []) + [f] 

	# If no groups are defined, every nodes have the same probability to be modified.
	if groups=={}: 
		groups_rec = {'all':dict_nodes_traj.keys()} 
		probabilities = {'all':1.}
	elif groups!={}:
	 	if not inplace:
			groups_rec = deepcopy(groups)
		else:
			groups_rec = groups
	
	# Remove all points in groups which are not crossed by any flights.
	for g, nodes in groups_rec.items():
		nodes_copy = nodes[:]
		for n in nodes_copy:
			if not n in dict_nodes_traj.keys(): groups_rec[g].remove(n)

	all_groups = groups_rec.keys()
	#print all_groups

	if sum(np.array([probabilities[gg] for gg in all_groups]))>0.:
		# If there is at least one group wiht proba>0, renormalize probas
		# Frozen list of probabilities for each group.
		probas_g = np.array([probabilities[gg] for gg in all_groups])
	else:
		if not hard_fixed:
			# Convert the remaining groups with 0 probabilities into groups with eaual probabilities.
			probas_g = np.array([1./len(all_groups) for gg in all_groups]) # If the last groups have zero probability they are fixed a non-zero probability again.
		else:
			# Leave the nodes in groups with 0 probabilities untouched.
			n_iter_max = 0

	## Frozen list of probabilities for each group.
	#probas_g = np.array([probabilities[gg] for gg in all_groups])

	eff_prev = eff/2.

	n_iter = 0
	while eff < eff_target and n_iter<n_iter_max:# and abs(eff - eff_prev)/eff>stop_criteria:
		# Choose one group with given probabilites for groups.
		g = choice(all_groups, p=probas_g)
		#print "g=", g, "; len(groups[g]) = ", len(groups[g])
		try:
			# Choose a node in the group with equal probabilities.
			n = choice(groups_rec[g])
		except ValueError:
			print "Group", g, "is empty, I remove it."
			del groups_rec[g]
			all_groups = groups_rec.keys()
			# Recompute probabilities for groups.
			if sum(np.array([probabilities[gg] for gg in all_groups]))>0.:
				# If there is at least one group wiht proba>0, renormalize probas
				probas_g = np.array([probabilities[gg] for gg in all_groups])/sum(np.array([probabilities[gg] for gg in all_groups])) # Recompute probabilities
			else:
				if not hard_fixed:
					# Convert the remaining groups with 0 probabilities into groups with equal probabilities.
					probas_g = np.array([1./len(all_groups) for gg in all_groups]) # If the last groups have zero probability they are fixed a non-zero probability again.
				else:
					# Leave the nodes in groups with 0 probabilities untouched.
					break
			if len(groups_rec)==0: 
				# No more increase in efficiency
				break
			else:
				continue

		#print "n=", n, "; len(dict_nodes_traj[n]) = ", len(dict_nodes_traj[n])

		# Choose a flight among those which cross this node with equal probabilities.
		f = choice(dict_nodes_traj[n])

		# Don't do anything if the trajectory is too short.
		if len(trajs_rec[f])<3:
			continue

		try:
			# Find the position of the point n in the trajectory of the flight.
			p = trajs_rec[f].index(n)
		except:
			print "flight:", f
			print "trajs_rec[f]", trajs_rec[f]
			raise

		# Coordinates of points before and after navpoint n.
		cc_before, cc_after = coords_func(trajs_rec[f][p-1]), coords_func(trajs_rec[f][p+1])
		# old length of trajectory between point before and point after.
 		old = dist_func([trajs_rec[f][p-1], trajs_rec[f][p]]) + dist_func([trajs_rec[f][p+1], trajs_rec[f][p]])
 		# New length (straight line).
 		new = dist_func([trajs_rec[f][p-1], trajs_rec[f][p+1]])

 		if new < old: # If the three points are not already aligned.
 			# There is room for optimization here...
 			#n = trajs_rec[f][p]
			#g = find_group(n, groups)
			if len(dict_nodes_traj[n])>1:
				# Remove the flight from the flights which cross the old point n.
				dict_nodes_traj[n].remove(f)
			else:
				# If it was the last flight of the list, delete the list and remove node from 
				# the group.
				del dict_nodes_traj[n]
				groups_rec[g].remove(n)
 			
 			# Two features: either remove old point or add a new point between the points.
 			if remove_nodes:
 				trajs_rec[f].remove(n)
 			else:
 				# Compute coordinates of middle point 
 				coords = [pl.mean([cc_before[0], cc_after[0]]), pl.mean([cc_before[1], cc_after[1]])]
 				new_node, trajs_rec, G = add_node_func(trajs_rec, G, coords, f, p)
 				dict_nodes_traj[new_node] = [f]
 				groups_rec[g].append(new_node)
			
			# Change the efficiency based on the new length of the trajectory.
			eff_prev = eff
			eff = S/(S/eff+ (new-old))

	#		print "eff=", eff

		n_iter += 1

	if n_iter == n_iter_max:	
		print "Hit maximum number of iterations"
	print "New efficiency (cumul):", eff
	print "New efficiency:", compute_efficiency(trajs_rec, dist_func=dist_func)[0]
	return trajs_rec, eff, G, groups_rec

# ==================== Obsolete ====================== #

def select_heigths(th):
	"""
	Sorts the altitude th increasingly, decreasingly or half/half at random.
	"""
	coin=rd.choice(['up','down','both'])
	if coin=='up' : th.sort()
	if coin=='down': th.sort(reverse=True)
	if coin=='both':
		a=th[:len(th)/2]
		a.sort()
		b=th[len(th)/2:]
		b.sort(reverse=True)
		th=a+b
	return th

def calculate_time(C, T, vm):
	"""
	Compute times of crossings given a trajectory C and first time T.
	"""
	t = [0]*len(C)
	t[0] = T
	for i in range(1, len(C)):
		dt = timedelta(seconds=dist([to_coord(C[i-1]), to_coord(C[i])])/vm)
		t[i] = t[i-1] + dt
	return t

def neighboors_traj(x):
	"""
	Build the list of "neighbors" for trajectories. Two trajectories are neighbors 
	if they intersect each other. Each trajectory checks the trajectories before
	itself in the list.
	"""
	#xp=[LineString([gall(b) for b in a]) for a in x]
	xp=[LineString(a) for a in x]
	neig=[[j for j in range(i) if xp[i].intersects(xp[j])] for i in range(len(xp))]
	
	return neig

def create_traffic(config, N_flights):
	"""
	Generate N_flights flights with distribution of altitude taken from 
	the file in config['type']. 
	TODO: use networkx instead of igraph.
	"""
	with open(config['type'],'r') as fr:
		x=fr.read()
	
	# Number of flights for the altitude distribution
	old_Nflight = int(x.split('\n')[0].split('\t')[0])

	# Put trajectories in a good format?
	x=[[[[float(b.split(',')[0]),float(b.split(',')[1])],float(b.split(',')[2]),datetime.strptime(b.split(',')[3],'%Y-%m-%d %H:%M:%S') ] for b in  a.split('\t')[2:-1] ] for a in x.split('\n')[1:-1]]
	
	# Compute velocities on each segment to have median velocity.
	v=[dist([gall(f[i][0]),gall(f[i+1][0])])/(f[i+1][2]-f[i][2]).total_seconds() for f in x for i in range(1,len(f)-1)]
	vm=pl.median(v)
		
	'Select from distribution OD for flights'
	Ap=[[a[0][0],a[-1][0]] for a in x]

	# Choose OD for new flights
	Ap=[rd.choice(Ap) for a in range(N_flights)]


	Aps=list(set([to_str([a[0]])+','+to_str([a[1]]) for a in Ap]))
	Aps=[[to_coord_ng(a.split(',')[0])  ,to_coord_ng(a.split(',')[1] ) ] for a in Aps]
	
	'Calculate angle between different OD'
	# This is for putting half flights on odd FL and the other half on even FLs.
	Ang=[mt.atan2(-(a[0][1]-a[1][1]),(a[0][0]-a[1][0]))  for a in [[gall(a[0]),gall(a[1])] for a in Ap]] 
	'Select from distribution height for fligths'
	h=[(int(b[1])/10)*10. for a in x for b in a if b[1]>=240.]
	hp=[a for a in h if a%20==0]
	hd=[a for a in h if a%20!=0]
	h=[hp,hd]
	'Choice different heights respect to the direction - odd rule'
	H=[rd.choice(h[Ang[i]<0]) for i in range(N_flights)] # Not used
	T=[rd.choice([a[0][2] for a in x]) for i in range(N_flights)]
	T=[tf_time(a) for a in T]
	
	# Navigation points
	nvp=list(set([str(b[0][0])+' '+str(b[0][1]) for a in x for b in a]))
	
	'Create Graph from Data'
	g=Graph(len(nvp))
	g.vs["name"]=nvp
	g.es["weight"] = 1.0
	for f in x:
		for i in range(len(f)-1):
			g[to_str(f[i]),to_str(f[i+1])]=dist([gall(f[i][0]),gall(f[i+1][0])])

	'Calculate the shortest path on the net between OD'
	to_str2 = lambda z: str(z[0])+' '+str(z[1])
	L=[g.shortest_paths(source=to_str2(a[0]),target=to_str2(a[1]),weights=g.es["weight"])[0][0] for a in Aps]
	#S=[dist([gall(a[0]),gall(a[1])]) for a in Aps]
	#S=pl.mean(S) # Srange...
	#Eff=pl.mean(S)/pl.mean(L)
	
	#print "Efficence",Eff,N_flights
	
	gV={i:g.vs["name"][i] for i in range(len(g.vs["name"]))}	
	
	P=[[gV[e] for e in g.get_shortest_paths(to_str2(a[0]),to=to_str2(a[1]),weights=g.es["weight"])[0]] for a in Aps]
	
	routes=[[to_coord(a) for a in b] for b in P]

	return routes, Ap, T

def get_originale_net(config, newEff, Nflight):

	routes, Ap, T = create_traffic(config, Nflight)
	
	routes = rectificate_trajectories(routes, newEff)
	
	#print to_OD_inv(routes[0])
	
	routes = {to_OD_inv(a):a for a in routes}
	routes = [routes[to_OD2(a)] for a in Ap]
	#~ NEW
	neig=neighboors_traj(routes)
			
	P=[[str(invgall(a)[0])+' '+str(invgall(a)[1]) for a in b] for b in routes]

	'Create routes'
	routes=[[] for i in range(len(P))]
	for i in range(len(P)):
		th=select_heigths([rd.choice(h[Ang[i]<0]) for j in range(len(P[i]))])
		t=[datetime.strftime(a,"%Y-%m-%d %H:%M:%S:%f") for a in calculate_time(P[i],T[i],float(vm))]
		routes[i]= zip(P[i],th,t)
		
	return routes,neig	

def print_net(net,T,neig):
	in_f = lambda x: 'inputABM_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'
	with open('OUTPUT_net_eff/'+in_f(T),'w') as fw:
		fw.write(str(len(net))+'\tNflight\n')
		for i in range(len(net)):
			fw.write(str(i+1)+'\t'+str(len(net[i]))+'\t')
			for p in net[i]:
				fw.write(p[0].replace(' ',',')+','+str(p[1])+','+p[2]+'\t')
			fw.write('\n')
	
	in_neig = lambda x: 'neighboors_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'
	
	with open('OUTPUT_net_eff/'+in_neig(T),'w') as fw:
		fw.write("\t\n".join(["\t".join([str(h) for h in neig[i]]) for i in range(len(neig))]))
			
def new_route(newEff,newNflight,nsim):

	config={
		'type':'../trajectories/trajectories.dat',
		'distr':'from_file',
		'coll_free': True
	}
	#config['type']='DATA/AIRAC_334/m1_acc_LIRR_'+str(i)+'.dat'
	net,neig=get_originale_net(config,newEff,newNflight)
	
	print_net(net,[newEff,newNflight,nsim],neig)

if __name__=='__main__':
	new_route(0.9928914418,1500,0)
