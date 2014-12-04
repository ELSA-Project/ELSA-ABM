#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter 
import pylab as pl
import random as rd
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

def rettifica(routes, Eff, D, eff_target):
	"""
	Given all routes and a value of efficiency, modify the trajectories 
	by creating a new point between two existing points chosen at random
	on a trajectory. Modifies trajectories until the target efficiency 
	is met.
	Args:
		routes: list of trajectories (list of points)
		Eff: initial value of efficiency
		D: direct path total length
		eff_target: target for efficiency
	Return:
		routes: modified routes.
	"""
	Nf=len(routes)

	while Eff<=eff_target:
		f=rd.choice(range(len(routes)))
		if len(routes[f])<3:
			continue
		p=rd.choice(range(1,len(routes[f])-1))

		old=dist([routes[f][p-1],routes[f][p]])+dist([routes[f][p+1],routes[f][p]])
		new=dist([routes[f][p-1],routes[f][p+1]])
		if new < old :
			routes[f][p][0]=pl.mean([routes[f][p-1][0],routes[f][p+1][0]])
			routes[f][p][1]=pl.mean([routes[f][p-1][1],routes[f][p+1][1]])
			Eff=D/(D/Eff+ ((new-old)/Nf))

	return routes

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

def get_originale_net(config, newEff, Nflight):
	"""
	Generate Nflight flights with distribution of altitude taken from 
	the file in config['type']. 
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
	Ap=[rd.choice(Ap) for a in range(Nflight)]


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
	'choice different height respect to the direction - odd rule'
	H=[rd.choice(h[Ang[i]<0]) for i in range(Nflight)]
	T=[rd.choice([a[0][2] for a in x]) for i in range(Nflight)]
	tf_time= lambda z: datetime.strptime("2010-06-02 0:0:0:0",'%Y-%m-%d %H:%M:%S:%f') +timedelta(seconds=((int((z - datetime.strptime("2010-06-02 0:0:0:0",'%Y-%m-%d %H:%M:%S:%f') ).total_seconds()) /4)*4))
	
	T=[tf_time(a) for a in T]
	
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
	S=[dist([gall(a[0]),gall(a[1])]) for a in Aps]
	S=pl.mean(S)
	Eff=pl.mean(S)/pl.mean(L)
	
	#print "Efficence",Eff,Nflight
	
	gV={i:g.vs["name"][i] for i in range(len(g.vs["name"]))}	
	
	P=[[gV[e] for e in g.get_shortest_paths(to_str2(a[0]),to=to_str2(a[1]),weights=g.es["weight"])[0]] for a in Aps]
	
	route=[[to_coord(a) for a in b] for b in P]
	route=rettifica(route,Eff,S,newEff)
	
	print to_OD_inv(route[0])
	
	route={to_OD_inv(a):a for a in route}
	route=[route[to_OD2(a)] for a in Ap]
	#~ NEW
	neig=neighboors_traj(route)
			
	P=[[str(invgall(a)[0])+' '+str(invgall(a)[1]) for a in b] for b in route]

	'Create route'
	route=[[] for i in range(len(P))]
	for i in range(len(P)):
		th=select_heigths([rd.choice(h[Ang[i]<0]) for j in range(len(P[i]))])
		t=[datetime.strftime(a,"%Y-%m-%d %H:%M:%S:%f") for a in calculate_time(P[i],T[i],float(vm))]
		route[i]= zip(P[i],th,t)
		
	return route,neig	

def print_net(net,T,neig):
	in_f = lambda x: 'inputABM_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'
	fw=open('OUTPUT_net_eff/'+in_f(T),'w')
	fw.write(str(len(net))+'\tNflight\n')
	for i in range(len(net)):
		fw.write(str(i+1)+'\t'+str(len(net[i]))+'\t')
		for p in net[i]:
			fw.write(p[0].replace(' ',',')+','+str(p[1])+','+p[2]+'\t')
		fw.write('\n')
	fw.close()
	
	in_neig = lambda x: 'neighboors_n-'+str(x[2])+'_Eff-'+str(x[0])+'_Nf-'+str(x[1])+'.dat'
	fw=open('OUTPUT_net_eff/'+in_neig(T),'w')
	fw.write("\t\n".join(["\t".join([str(h) for h in neig[i]]) for i in range(len(neig))]))
	fw.close()
		
def new_route(newEff,newNflight,nsim):

	config={
		'type':'../trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat',
		'distr':'from_file',
		'coll_free': True
	}
	#config['type']='DATA/AIRAC_334/m1_acc_LIRR_'+str(i)+'.dat'
	net,neig=get_originale_net(config,newEff,newNflight)
	
	print_net(net,[newEff,newNflight,nsim],neig)

if __name__=='__main__':
	new_route(0.9928914418,1500,0)
