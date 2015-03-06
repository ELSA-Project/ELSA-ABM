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

# DIR='../trajectories/M1/'
# DIR3='../trajectories/M3_nodir/'

main_dir = os.path.abspath(__file__)
main_dir = os.path.split(os.path.dirname(main_dir))[0]

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

def n_actions_rerouting((m1, m3), thre=10):
	n_acts = 0
	for i in range(min(len(m1), len(m3))):
		x1, y1 = gall_pet(m1[i])
		x3, y3 = gall_pet(m3[i])
		if pl.sqrt((x1-x3)**2 + (y1-y3)**2)>thre:
			n_acts += 1
	return n_acts

def n_actions_tot((m1, hm1, m3, hm3), thre=10):
	n_acts = 0
	for i in range(min(len(m1), len(m3))):
		x1, y1 = gall_pet(m1[i])
		x3, y3 = gall_pet(m3[i])
		if pl.sqrt((x1-x3)**2 + (y1-y3)**2)>thre or abs(int(hm1[i]) - int(hm3[i]))>1:
			n_acts += 1
	return n_acts

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
	p = Pool(2)

	force = True

	temp = __import__(sys.argv[1], globals(), locals(), ['all_paras'], -1)
	
	all_paras = temp.all_paras

	with open(main_dir + '/trajectories/files/' + sys.argv[1] + '_files.pic', 'r') as f:
		files = pickle.load(f)

	for idx, (inpt, outpt) in enumerate(files):
		counter(idx, len(files), message="Computing differences between trajectories ... ")

		LH_file = main_dir + '/trajectories/metrics/L_H_' + outpt.split('/')[-1]
		if not os.path.exists(LH_file) or force:
			#F3 = main_dir + '/trajectories/M3/trajs_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '_0.dat'
			#F1 = main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'
			#print inpt
			#print outpt
			m1, hm1 = get_M(inpt)
			m3, hm3 = get_M(outpt)
			# print len(m1.keys())
			# print
			# print len(m3.keys())
			# raise Exception()
			# have a look to the fact that m1 and m3 do not have the same number of flights.
			L = p.map(lleng,[(m1[a], m3[a]) for a in m3 if a in m1.keys()])
			H = sum( p.map(hg,[(m1[a], m3[a], hm1[a], hm3[a]) for a in m1 if a in m3.keys()]))
			#NA = p.map(n_actions, [(m1[a], m3[a]) for a in m3 if a in m1.keys()])
			NA = p.map(n_actions_tot, [(m1[a], hm1[a], m3[a], hm3[a]) for a in m3 if a in m1.keys()])
			# L = p.map(lleng,[(m1[a],m3[a]) for a in m3])
			# H = sum( p.map(hg,[(m1[a],m3[a],hm1[a],hm3[a])for a in m1]))
			# NA = p.map(n_actions, [(m1[a],m3[a]) for a in m3])

			with open(LH_file, 'w') as f:
				pickle.dump({'L':L, 'H':H, 'NA':NA}, f)

	# n_iter = 100

	# force = True

	# main_dir = os.path.abspath(__file__)
	# main_dir = os.path.split(os.path.dirname(main_dir))[0]
	# zone = 'LIRR'

	# os.system('mkdir -p ' +  main_dir + '/trajectories/metrics/')

	# #F, F3, idxs = [], [], []
	# #print 'Computing differences between trajectories... '
	# sig_V_iter = [0.] + [10**(-float(i)) for i in range(5, -1, -1)]
	# t_w_iter = [40, 80, 120, 160, 240]
	# for sig_V in sig_V_iter:
	# 	for t_w in t_w_iter:
	# 		for i in range(n_iter):
	# 			counter(i, n_iter, message="Computing differences between trajectories for sig_V=" + str(sig_V) + " and t_w=" + str(t_w) + " ... ")
	# 			rep = main_dir + '/trajectories/metrics/L_H_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '.dat'
					
	# 			if not os.path.exists(rep) or force:
	# 				F3 = main_dir + '/trajectories/M3/trajs_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '_0.dat'
	# 				F1 = main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'

	# 				m1, hm1 = get_M(F1)
	# 				m3, hm3 = get_M(F3)

	# 				L = p.map(lleng,[(m1[a],m3[a]) for a in m3])
	# 				H = sum( p.map(hg,[(m1[a],m3[a],hm1[a],hm3[a])for a in m1]))
	# 				NA = p.map(n_actions, [(m1[a],m3[a]) for a in m3])

	# 				with open(rep, 'w') as f:
	# 					pickle.dump({'L':L, 'H':H, 'NA':NA}, f)



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
