#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter 
import pylab as pl
import random as rd
import numpy as np
from numpy.random import choice
import math as mt
from shapely.geometry import LineString

from igraph import *

from datetime import datetime,timedelta

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


__version__ = "1.3"

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

def compute_efficiency(trajectories, dist_func = dist):
	"""
	Compute the efficiency of a set of trajectories.
	"""

	L = [sum([dist_func([trajectories[f][p-1],trajectories[f][p]]) for p in range(1, len(trajectories[f]))]) for f in range(len(trajectories))]
	S = [dist_func([trajectories[f][0],trajectories[f][-1]]) for f in range(len(trajectories))]
	#L=[g.shortest_paths(source=to_str2(a[0]), target=to_str2(a[1]), weights=g.es["weight"])[0][0] for a in Aps]
	#S=[dist([gall(a[0]),gall(a[1])]) for a in Aps]
	return pl.mean(S)/pl.mean(L), pl.mean(S)

def rectificate_trajectories(trajs, eff_target, add_node_func = None, dist_func = dist, coords_func = lambda x: x, n_iter_max = 1000000, G=None, groups = {}, probabilities = {}, group_new_nodes = None):
	"""
	Given all trajectories and a value of efficiency, modify the trajectories 
	by creating a new point between two existing points chosen at random
	on a trajectory. Modifies trajectories until the target efficiency 
	is met.

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

	Return:
		trajs: modified trajectories.

	Changed in 1.1: efficiency is computed internally. Stop criteria on efficiency is < instead of <=.
		Added kwarg dist_func.
	Changed in 1.2: added dist_func, coords_func, n_iter_max, add_node_func, G.
	Changed in 1.3: added probabilities. Broken the legacy with coordinatesbased tarjectories.
	"""

	def add_node_func_coords(trajs, G, coords, f, p, groups, group_new_nodes, dict_nodes_traj):
		dict_nodes_traj[tuple(trajs[f][p])].remove(f) # PROBLEM

		trajs[f][p][0] = coords[0]
		trajs[f][p][1] = coords[1]

		groups[group_new_node].append(tuple(coords))
		
		dict_nodes_traj[tuple(coords)] = [f]
		return trajs, G, groups, dict_nodes_traj

	if add_node_func==None: add_node_func = add_node_func_coords

	print "Rectificating trajectories..."
	eff, S = compute_efficiency(trajs, dist_func = dist_func)
	print "Old efficiency:", eff
	Nf = len(trajs)

	dict_nodes_traj = {}
	for f in range(Nf):
		for p in trajs[f][1:-1]:
			#if p in dict_nodes_traj.keys():
			dict_nodes_traj[p] = dict_nodes_traj.get(p, []) + [f] # to each point, associates the list of flights going through.
			if len(dict_nodes_traj[p]) == 0:
				print "PROBLEM", f, p

	if groups=={}: 
		groups['all'] = dict_nodes_traj.keys() 
		group_new_nodes = 'all'
		probabilities['all'] = 1.

		#proba_f = [sum([probabilities[trajs[f][i]] for i in range(1,len(trajs[f])-1)]) for f in range(len(trajs))]
		#proba_f = np.array(proba_f)/sum(proba_f)
	#else:
		#proba_f = None
	all_groups = groups.keys()
	#assert not "new_nodes" in groups.keys()
	#groups['new_nodes'] = []
	#probabilities['new_nodes'] = 1./(1. + len(groups))
	probas_g = np.array([probabilities[gg] for gg in all_groups])
	#probas_g = probas_g/sum(probas_g)

	n_iter = 0
	while eff < eff_target or n_iter>n_iter_max:
		g = choice(all_groups, p=probas_g)
		#print "g=", g, "; len(groups[g]) = ", len(groups[g])
		n = choice(groups[g])
		#print "n=", n, "; len(dict_nodes_traj[n]) = ", len(dict_nodes_traj[n])
		f = choice(dict_nodes_traj[n])

		p = trajs[f].index(n)

		#f = choice(range(len(trajs)), p=proba_f)
		if len(trajs[f])<3:
			continue
		#if probabilities!={}:
		#	proba = [probabilities[trajs[f][i]] for i in range(1,len(trajs[f])-1)]
		#	proba = np.array(proba)/sum(proba)
		#else:
		#	proba = None
		#p = rd.choice(range(1,len(trajs[f])-1))
		#p = choice(range(1,len(trajs[f])-1), p=proba)

		cc_before, cc_after = coords_func(trajs[f][p-1]), coords_func(trajs[f][p+1])
 		old = dist_func([trajs[f][p-1], trajs[f][p]]) + dist_func([trajs[f][p+1], trajs[f][p]])
 		new = dist_func([trajs[f][p-1], trajs[f][p+1]])
 		if new < old :
 			coords = [pl.mean([cc_before[0], cc_after[0]]), pl.mean([cc_before[1], cc_after[1]])]
 			trajs, G, groups, dict_nodes_traj = add_node_func(trajs, G, coords, f, p, groups, group_new_nodes, dict_nodes_traj)
			eff=S/(S/eff+ ((new-old)/Nf))
			 # Remove the flight point from the list of this point.

		n_iter += 1

	print "New efficiency:", eff
	return trajs

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
	H=[rd.choice(h[Ang[i]<0]) for i in range(N_flights)]
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
