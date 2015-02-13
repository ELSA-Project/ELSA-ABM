#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')

from collections import Counter 
import pylab as pl
import random as rd
import numpy as np
from numpy.random import choice
import math as mt
from shapely.geometry import LineString

from igraph import *

from datetime import datetime,timedelta

from abm_strategic.utilities import select_interesting_navpoints, OD
from libs.tools_airports import build_long_2d
from libs.general_tools import insert_list_in_list

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


__version__ = "1.4"


def uniform_rectification():
	pass

def iter_partial_rectification(trajectories, eff_targets, G, metric='centrality', N_per_sector=1, **kwargs_rectificate):
	"""
	Used to iterate a partial_rectification without recomputing the best nodes each time.
	"""
	# Make groups
	n_best = select_interesting_navpoints(G, OD=OD(trajectories), N_per_sector=N_per_sector, metric=metric) # Selecting points with highest betweenness centrality within each sector
	n_best = [n for sec, points in n_best.items() for n in points]

	groups = {"C":[], "N":[]} # C for "critical", N for "normal"
	for n in G.G_nav.nodes():
		if n in n_best:
			groups["C"].append(n)
		else:
			groups["N"].append(n)
	probabilities = {"C":0., "N":1.} # Fix nodes with best score (critical points).

	final_trajs_list, final_eff_list, final_G_list, final_groups_list = [], [], [], []
	for eff_target in eff_targets:
		final_trajs, final_eff, final_G, final_groups = rectificate_trajectories_network(trajectories, eff_target, G.G_nav, groups=groups, probabilities=probabilities,\
			remove_nodes=True, **kwargs_rectificate)
		for new_el, listt in [(final_trajs, final_trajs_list), (final_eff, final_eff_list), (final_G, final_G_list), (final_groups, final_groups_list)]:
			listt.append(new_el)

	return final_trajs_list, final_eff_list, final_G_list, final_groups_list

def partial_rectification(trajectories, eff_target, G, metric='centrality', N_per_sector=1, **kwargs_rectificate):
	"""
	High level function for rectification. Fix completely N_per_sector points with 
	highest metric value per sector.
	"""
	# Make groups
	n_best = select_interesting_navpoints(G, OD=OD(trajectories), N_per_sector=N_per_sector, metric=metric) # Selecting points with highest betweenness centrality within each sector
	n_best = [n for sec, points in n_best.items() for n in points]

	groups = {"C":[], "N":[]} # C for "critical", N for "normal"
	for n in G.G_nav.nodes():
		if n in n_best:
			groups["C"].append(n)
		else:
			groups["N"].append(n)
	probabilities = {"C":0., "N":1.} # Fix nodes with best score (critical points).
	
	final_trajs, final_eff, final_G, final_groups = rectificate_trajectories_network(trajectories, eff_target, G.G_nav, groups=groups, probabilities=probabilities,\
		remove_nodes=True, **kwargs_rectificate)

	return final_trajs, final_eff, final_G, final_groups

def compute_efficiency(trajectories, dist_func = dist):
	"""
	Compute the efficiency of a set of trajectories.
	"""

	L = [sum([dist_func([trajectories[f][p-1],trajectories[f][p]]) for p in range(1, len(trajectories[f]))]) for f in range(len(trajectories))]
	S = [dist_func([trajectories[f][0],trajectories[f][-1]]) for f in range(len(trajectories))]
	#L=[g.shortest_paths(source=to_str2(a[0]), target=to_str2(a[1]), weights=g.es["weight"])[0][0] for a in Aps]
	#S=[dist([gall(a[0]),gall(a[1])]) for a in Aps]
	return sum(S)/sum(L), sum(S)

def find_group(element, groups):
	"""
	Beware: use b = {c:g for g in a.keys() for c in a[g]}
	"""
	for g, els in groups.items():
		for el in els:
			if el == element:
		#if el == element: break
				return g

def rectificate_trajectories_network(trajs, eff_target,	G, groups={}, probabilities={}, n_iter_max=1000000, remove_nodes=False, hard_fixed=True):
	"""
	Wrapper, to use with a network.
	"""

	def get_coords(nvp):
		return G.node[nvp]['coord']

	def add_node(trajs, G, coords, f, p):
		new_node = len(G.nodes())
		G.add_node(new_node, coord = coords)

		#trajs[f].remove(n)
		trajs[f][p] = new_node

		return new_node, trajs, G
		
	def d((n1, n2)):
		p1 = get_coords(n1)
		p2 = get_coords(n2)
		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

	trajs_old = trajs[:]
	trajs_rec, eff, G, groups_rec = rectificate_trajectories(trajs, eff_target, add_node_func=add_node, dist_func=d, coords_func=get_coords, \
		n_iter_max=n_iter_max, G=G, groups=groups, probabilities=probabilities, remove_nodes=remove_nodes, hard_fixed=hard_fixed)

	if remove_nodes: # Resample trajectories
		for i in range(len(trajs_old)):
			n_old = len(trajs_old[i])
			n_new = len(trajs_rec[i])
			n_gen = n_old - n_new

			long_dis, long_dis_cul = build_long_2d([G.node[n]['coord'] for n in trajs_rec[i]]) # Compute linear distance after each point

			long_dis_new_points = [(j+1.)*(long_dis_cul[-1]-long_dis_cul[0])/float(n_gen+1.) for j in range(n_gen)] # regularly spaced points along the trajectories.

			new_point_coords = []
			new_point_indices = []
			for d in long_dis_new_points:
				point_before = long_dis_cul[d>long_dis_cul].index(True) # Index of point before future point

				dn = np.array(trajs_rec[point_before+1]) - np.array(trajs_rec[point_before]) # direction vector
				dn = dn/np.norm(dn)
				new_point_coords.append(np.array(trajs_rec[point_before]) + (d-long_dis_cul[point_before])*dn)
				new_point_indices.append(point_before)

			# Change the trajectory
			trajs_rec[i] = insert_list_in_list(trajs_rec[i], new_point_coords, new_point_indices)

			# Add to network
			for coords in new_point_coords:
				G.add_node(len(G.nodes()), coord = coords)

	return trajs_rec, eff, G, groups_rec

def rectificate_trajectories(trajs, eff_target, G=None, groups={}, add_node_func=None, dist_func=dist, coords_func=lambda x: x, \
	n_iter_max=1000000, probabilities={}, remove_nodes=False, hard_fixed=False, stop_criteria=0.0001):
	"""
	Given all trajectories and a value of efficiency, rectifiy the trajectories. The rectification can be 
	done in two ways: either some intermediate navpoints are removed from the trajectories,	or they are 
	moved to another position which straightens up the segment. The points are gathered in groups defined by the 
	user with kwarg 'groups'. These groups have different probabilities of being chosen. Then a navpoint is 
	chosen in this group, and finally a flight which crosses the navpoint is chosen. This goes on until the 
	target efficiency is met or or the stop criteria are met.

	Args:
		trajs: list of trajectories (list of points)
		eff_target: target for efficiency

	Kwargs:
		add_node_func: function which encode how to add the new nodes. 
			Signature: trajectories, network, coordinates of the new point, index of flight, index of p in trajectory of flight
		dist_func: function which associates a distance to a couple of points.
			Signature: 2-tuple of points from trajectories.
		coords_func: function which associates some coordinates to a point in a trajectory.
			Signature: point from a trajectory.
		n_iter_max: maximum number of iteration to reach target efficiency.
		G: network object which could contain coordinates of points.
		groups: grouping of nodes which don't have the same probability of being rectified.
		probabilities: dictionnary with the groups as keys and the probabilities of all points of being rectified as values.
		remove_nodes: if True, remove the chosen nodes from the trajectories. Otherwise, duplicate and move the existing point.
		hard_fixed: if False, groups having 0 probability of being chose with switch to non-zero probablity once every other 
					groups are empty. Other wise, exit the function when all groups are empty or with zero probability.
		stop_criteria: rectification stops when the relative change in efficiency between two rounds falls below this threshold.

 
	Return:
		trajs: modified trajectories.
		eff: corresponding efficiency.
		G: networkx graph with added or removed nodes.
		groups: the final groups.

	Changed in 1.1: efficiency is computed internally. Stop criteria on efficiency is < instead of <=.
		Added kwarg dist_func.
	Changed in 1.2: added dist_func, coords_func, n_iter_max, add_node_func, G.
	Changed in 1.3: added probabilities. Broken the legacy with coordinates-based tarjectories. Added groups, remove_nodes, hard_fixed.
	Changed in 1.4: added stop_criteria.
	"""

	def add_node_func_coords(trajs, G, coords, f, p):
		trajs[f][p][0] = coords[0]
		trajs[f][p][1] = coords[1]

		return coords, trajs, G

	if add_node_func==None: 
		add_node_func = add_node_func_coords

	print "Rectificating trajectories..."
	eff, S = compute_efficiency(trajs, dist_func=dist_func)
	print "Initial efficiency/target efficiency:", eff, '/', eff_target
	Nf = len(trajs)

	# To each point, associate the list of flights going through.
	dict_nodes_traj = {}
	for f in range(Nf):
		for p in trajs[f][1:-1]:
			dict_nodes_traj[p] = dict_nodes_traj.get(p, []) + [f] 

	# If no groups are deinfed, every nodes have the same probability to be modified.
	if groups=={}: 
		groups['all'] = dict_nodes_traj.keys() 
		probabilities['all'] = 1.
	
	# Remove all points in groups which are not crossed by any flights.
	for g, nodes in groups.items():
		nodes_copy = nodes[:]
		for n in nodes_copy:
			if not n in dict_nodes_traj.keys(): groups[g].remove(n)

	all_groups = groups.keys()

	# Frozen list of probabilities for each group.
	probas_g = np.array([probabilities[gg] for gg in all_groups])

	eff_prev = eff/2.

	n_iter = 0
	while eff < eff_target and n_iter<n_iter_max and abs(eff - eff_prev)/eff<stop_criteria:
		# Choose one group with given probabilites for groups.
		g = choice(all_groups, p=probas_g)
		#print "g=", g, "; len(groups[g]) = ", len(groups[g])
		try:
			# Choose a node in the group with eauql probabilities.
			n = choice(groups[g])
		except ValueError:
			print "Group", g, "is empty, I remove it."
			del groups[g]
			all_groups = groups.keys()
			# Recompute probabilities for groups.
			if sum(np.array([probabilities[gg] for gg in all_groups]))>0.:
				# If there is at least one group wiht proba>0, renormalize probas
				probas_g = np.array([probabilities[gg] for gg in all_groups])/sum(np.array([probabilities[gg] for gg in all_groups])) # Recompute probabilities
			else:
				if not hard_fixed:
					# Convert the remaining groups with 0 probabilities into groups with eaual probabilities.
					probas_g = np.array([1./len(all_groups) for gg in all_groups]) # If the last groups have zero probability they are fixed a non-zero probability again.
				else:
					# Leave the nodes in groups with 0 probabilities untouched.
					break
			if len(groups)==0: 
				# No more increase in efficiency
				break
			else:
				continue

		#print "n=", n, "; len(dict_nodes_traj[n]) = ", len(dict_nodes_traj[n])

		# Choose a flight among those which cross this node with equal probabilities.
		f = choice(dict_nodes_traj[n])

		# Don't do anything if the trajectory in too short.
		if len(trajs[f])<3:
			continue

		try:
			# Find the position of the point n in the trajectory of the flight.
			p = trajs[f].index(n)
		except:
			print "flight:", f
			print "trajs[f]", trajs[f]
			raise

		# Coordinates of points before and after navpoint n.
		cc_before, cc_after = coords_func(trajs[f][p-1]), coords_func(trajs[f][p+1])
		# old length of trajectory between point before and point after.
 		old = dist_func([trajs[f][p-1], trajs[f][p]]) + dist_func([trajs[f][p+1], trajs[f][p]])
 		# New length (straight line).
 		new = dist_func([trajs[f][p-1], trajs[f][p+1]])

 		if new < old: # If the three points are not already aligned.
 			# There is room for optimization here...
 			#n = trajs[f][p]
			#g = find_group(n, groups)
			if len(dict_nodes_traj[n])>1:
				# Reomve the flight from the flights which cross the old point n.
				dict_nodes_traj[n].remove(f)
			else:
				# If it was the last flight of the list, delete the list and remove node from 
				# the group.
				del dict_nodes_traj[n]
				groups[g].remove(n)
 			
 			# Two features: either remove old point or add a new point between the points.
 			if remove_nodes:
 				trajs[f].remove(n)
 			else:
 				# Compute coordinates of midlle point 
 				coords = [pl.mean([cc_before[0], cc_after[0]]), pl.mean([cc_before[1], cc_after[1]])]
 				new_node, trajs, G = add_node_func(trajs, G, coords, f, p)
 				dict_nodes_traj[new_node] = [f]
 				groups[g].append(new_node)
			
			# Change the efficiency based on the new length of the trajectory.
			eff_prev = eff
			eff=S/(S/eff+ (new-old))

		n_iter += 1

	if n_iter == n_iter_max:	
		print "Hit maximum number of iterations"
	print "New efficiency:", eff
	return trajs, eff, G, groups

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
