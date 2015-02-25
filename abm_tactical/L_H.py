#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')

from multiprocessing import Pool
import shelve
import os
import pylab as pl
from shapely.geometry import LineString,Point,Polygon
import scipy.stats as st
import pickle

from libs.general_tools import counter

DIR='../trajectories/M1/'
DIR3='../trajectories/M3_nodir/'

def save_dump(filename, varib = None ):
	my_shelf = shelve.open(filename,'n') # 'n' for new

	if varib == None:
		for key in dir():
			try:
				my_shelf[key] = globals()[key]
			except TypeError:
				#
				# __builtins__, my_shelf, and imported modules can not be shelved.
				#
				print('ERROR shelving: {0}'.format(key))
			except NotImplementedError:
				print('ERROR shelving: {0}'.format(key))
	else:
		for key in varib:
			try:
				my_shelf[key] = globals()[key]
			except TypeError:
				#
				# __builtins__, my_shelf, and imported modules can not be shelved.
				#
				print('ERROR shelving: {0}'.format(key))
			except NotImplementedError:
				print('ERROR shelving: {0}'.format(key))
			
	my_shelf.close()

def readc(file_r):
    fr=open(file_r,'r')
    x=fr.read()
    fr.close()
    return x

dist = lambda x,y: pl.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
 
def hg((m1,m3,hm1,hm3)):
	return int(len(m1)==len(m3) and hm1!=hm3)

leng = lambda x: LineString(map(gall_pet, x)).length

def lleng((m1,m3)):
    return leng(m1) - leng(m3)

def different((m1,hm1,m3,hm3)):
    if len(m1)!=len(m3): return 1
    if hm1!=hm3: return 1
    m1=map(gall_pet,m1)
    m3=map(gall_pet,m3)
    if max((dist(m1[i],m3[i]) for i in xrange(len(m1)))) < 1: return 0
    else: return 1

def get_M(file_r):
    m1=readc(file_r)
    m1={a.split('\t')[0]:map(lambda x:x.split(','),a.split('\t')[2:-1]) for a in m1.split('\n')[1:-1]}
    Fl = m1.keys()
    #t={f:[to_dat(a[3]) for a in m1[f]] for f in Fl}
    hei = {f:[a[2] for a in m1[f]] for f in Fl}
    m1={f:[tuple(map(float,a[:2])) for a in m1[f]] for f in Fl}
    #nvp=list(set([a for f in Fl for a in m1[f]]))
    
    return m1,hei

gall_pet = lambda x :  ( 6371000.*pl.pi*x[1]/ (180.*pl.sqrt(2)) , 6371000.*pl.sqrt(2)*pl.sin(pl.pi*(x[0]/180.)) )
  
def to_P(m1):
    return [LineString(map(gall_pet,m1[f])) for f in m1]

def best_p(x):
    return Point(x.coords[0]).distance(Point(x.coords[-1]))  
 
def get_eff(sim):
    m3=get_M(DIR+sim)[0]
    m3=to_P(m3)   
    return sum(map(best_p,m3))/float(sum(map(lambda x: x.length,m3)))


if __name__=='__main__':
	p = Pool(1)

	n_iter = 10

	main_dir = os.path.abspath(__file__)
	main_dir = os.path.split(os.path.dirname(main_dir))[0]
	zone = 'LIRR'

	os.system('mkdir -p ' +  main_dir + '/trajectories/metrics/')

	#F, F3, idxs = [], [], []
	print 'Computing differences between trajectories... '
	sig_V_iter = [0.] + [10**(-float(i)) for i in range(5, -1, -1)]
	t_w_iter = [40, 80, 120, 160, 240]
	for sig_V in sig_V_iter:
		for t_w in t_w_iter:
			for i in range(n_iter):
				F3 = main_dir + '/trajectories/M3/trajs_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '_0.dat'
				F1 = main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'

				m1, hm1 = get_M(F1)
				m3, hm3 = get_M(F3)
				
				L = p.map(lleng,[(m1[a],m3[a]) for a in m3])
				H = sum( p.map(hg,[(m1[a],m3[a],hm1[a],hm3[a])for a in m1]))

				rep = main_dir + '/trajectories/metrics/L_H_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '.dat'
				with open(rep, 'w') as f:
					pickle.dump({'L':L, 'H':H}, f)

	# F=[a for a in os.listdir(DIR) if 'direct' in a and 'stats' not in a and 'eff' not in a]
		
	# M1={r:[] for r in  list(set(map(lambda x: x.split('_')[6],F)))}
	# for f in F:
	# 	M1[f.split('_')[6]].append(f)
		
	# #F3=[a for a in os.listdir(DIR3) if 'direct' in a and 'stats' not in a and 'eff' not in a]
	# M3={r:[] for r in  list(set(map(lambda x: x.split('_')[6],F3)))}
	# for f in F3:
	# 	M3[f.split('_')[6]].append(f)
	

	# L={acc:[] for acc in M1}
	# H={acc:[] for acc in M1}

	# for acc in M1:
	# 	print acc
	# 	for sim in M1[acc]:
	

	#for i in range(len(F)):
		

	

	#save_dump('../DATA/len_rer_nodir.dump',['L','H'])
