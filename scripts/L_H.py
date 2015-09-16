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
import os.path, time
import datetime as dt
import numpy as np

from libs.general_tools import counter

import abm_strategic
result_dir = abm_strategic.result_dir

# DIR='../trajectories/M1/'
# DIR3='../trajectories/M3_nodir/'

main_dir = os.path.abspath(__file__)
main_dir = os.path.split(os.path.dirname(main_dir))[0]

class Reached_N_Files(Exception):
	pass

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

#leng = lambda x: LineString(map(gall_pet, x)).length

def leng(x):
	#print x
	return LineString(map(gall_pet, x)).length

#def lleng((m1,m3)):
def lleng((m1, m3)):
	# m1, m3 = xx
	# try:
	# 	x = leng(m1)
	# except:
	# 	print m1
	# 	print "pouet"
	# 	raise
	# try:
	# 	x = leng(m3)
	# except:
	# 	print m1
	# 	print m3
	# 	print "pouet3"
	# 	raise
	if len(m1)>1 and len(m3)>1:
		return leng(m1) - leng(m3)
	else:
		return 0

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
	# This works well for the output of the model but
	# not for real trajectories. Indeed, the former 
	# have exactly one m3 point per action of the 
	# controller after the first deviation. 
	n_acts = 0
	for i in range(min(len(m1), len(m3))):
		x1, y1 = gall_pet(m1[i])
		x3, y3 = gall_pet(m3[i])
		if pl.sqrt((x1-x3)**2 + (y1-y3)**2)>thre or abs(int(hm1[i]) - int(hm3[i]))>1:
			n_acts += 1
	return n_acts

def time_flight(tm1):
	return (tm1[-1] - tm1[0]).total_seconds()

def delay_flight((tm1, tm3)):
	if len(tm1)>1 and len(tm3)>1:
		return (tm3[-1] - tm1[-1]).total_seconds()
	else:
		return 0.
def get_M(file_r):
	m1=readc(file_r)
	m1={a.split('\t')[0]:map(lambda x:x.split(','),a.split('\t')[2:-1]) for a in m1.split('\n')[1:-1]}
	Fl = m1.keys()
	#t={f:[to_dat(a[3]) for a in m1[f]] for f in Fl}
	hei = {f:[a[2] for a in m1[f]] for f in Fl}
	m1={f:[tuple(map(float,a[:2])) for a in m1[f]] for f in Fl}
	#nvp=list(set([a for f in Fl for a in m1[f]]))

	return m1,hei

def get_M_bis(file_r, put_m1_date=None):
	m1=readc(file_r)
	m1={a.split('\t')[0]:map(lambda x:x.split(','),a.split('\t')[2:-1]) for a in m1.split('\n')[1:-1]}
	Fl = m1.keys()
	t={f:[to_datetime(a[3], put_m1_date=put_m1_date) for a in m1[f]] for f in Fl}
	hei = {f:[a[2] for a in m1[f]] for f in Fl}
	m1={f:[tuple(map(float,a[:2])) for a in m1[f]] for f in Fl}
	#nvp=list(set([a for f in Fl for a in m1[f]]))

	return m1, hei, t

def to_datetime(st, put_m1_date=None):
	date_st, hour_st = st.split(' ')
	year, month, day = date_st.split('-')
	coin = hour_st.split(':')
	if len(coin)==3:
		hour, minutes, sec = coin
	else:
		hour, minutes, sec, ms = coin

	hour = int(hour)
	if put_m1_date!=None:
		year, month, day = put_m1_date
	else:
		day = int(day)
		year = int(year)
		month = int(month)

	if int(hour)>23:
		day += 1
		hour -= 24
	try:
		ddt =  dt.datetime(int(year), int(month), int(day), int(hour), int(minutes), int(sec))#, int(ms))
	except:
		print st 
		raise
	return ddt

gall_pet = lambda x :  ( 6371000.*pl.pi*x[1]/ (180.*pl.sqrt(2)) , 6371000.*pl.sqrt(2)*pl.sin(pl.pi*(x[0]/180.)) ) #Input format: lat/lon

def to_P(m1):
	return [LineString(map(gall_pet,m1[f])) for f in m1]

def best_p(x):
	return Point(x.coords[0]).distance(Point(x.coords[-1]))

def get_eff(sim):
	m3=get_M(DIR+sim)[0]
	m3=to_P(m3)
	return sum(map(best_p,m3))/float(sum(map(lambda x: x.length,m3)))

def extract_from_file(files):
	pass

if __name__=='__main__':
	p = Pool(1)

	n_files_to_analyse = -1

	force = False

	temp = __import__(sys.argv[1], globals(), locals(), ['all_paras'], -1)
	
	all_paras = temp.all_paras

	with open(result_dir + '/trajectories/files/' + sys.argv[1] + '_files.pic', 'r') as f:
		files = pickle.load(f)

	#t_now = dt.now()
	print len(files)
	try:
		for idx, (inpt, outpt) in enumerate(files):
			#print outpt 
			# n = 4
			# allowed = ['/home/earendil/Documents/ELSA/ABM/results/trajectories/M1/trajs_Real_LI_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed_' + str(i) + '.dat' for i in range(n)]
			# if inpt in allowed:
			max_it = min(n_files_to_analyse, len(files)) if n_files_to_analyse>0 else len(files)
			counter(idx, max_it, message="Computing differences between trajectories ... ")

			#t = dt.os.path.getmtime(outpt)
			#print "last modified: %s" % time.ctime()
			#raise Exception()
			#print "created: %s" % time.ctime(os.path.getctime(outpt))

			LH_file = result_dir + '/trajectories/metrics/L_H_' + outpt.split('/')[-1]
			if not os.path.exists(LH_file) or force:

				#print inpt
				#print outpt
				#F3 = result_dir + '/trajectories/M3/trajs_' + zone + '_real_data_sigV' + str(sig_V) + '_t_w' + str(t_w) + '_' + str(i) + '_0.dat'
				#F1 = result_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'
				try:
					if n_files_to_analyse>0:
						assert idx<n_files_to_analyse
					m1, hm1, tm1 = get_M_bis(inpt)
					m3, hm3, tm3 = get_M_bis(outpt, put_m1_date=(2010, 5, 6))
					#print "pouet"
					#print len(m1), len(m3)
					#print 
					# TODO: have a look to the fact that m1 and m3 do not have the same number of flights.
					L = p.map(lleng,[(m1[a], m3[a]) for a in sorted(m3) if a in m1.keys()])
					#print [len(xx) for xx in [(m1[a], m3[a]) for a in sorted(m3) if a in m1.keys()]]
					#L = np.vectorize(lleng)([(m1[a], m3[a]) for a in sorted(m3) if a in m1.keys()])
					#L = [lleng((m1[a], m3[a])) for a in sorted(m3) if a in m1.keys()]
					H = sum( p.map(hg,[(m1[a], m3[a], hm1[a], hm3[a]) for a in sorted(m1) if a in m3.keys()]))
					#NA = p.map(n_actions, [(m1[a], m3[a]) for a in m3 if a in m1.keys()])
					NA = p.map(n_actions_tot, [(m1[a], hm1[a], m3[a], hm3[a]) for a in sorted(m3) if a in m1.keys()])
					T = p.map(time_flight, [tm1[a] for a in sorted(m3) if a in m1.keys()])
					dT = p.map(delay_flight, [(tm1[a], tm3[a]) for a in sorted(m3) if a in m1.keys()])
				except IOError:
					print "Did not find files:"
					print (inpt, outpt)
				except AssertionError:
					raise Reached_N_Files()
				finally:
					with open(LH_file, 'w') as f:
						pickle.dump({'L':L, 'H':H, 'NA':NA, 'N_flights':len(m1), 'T':T, 'dT':dT}, f)
	
	except Reached_N_Files:
		print
		print "Reached the number of files required, analyzed only", n_files_to_analyse, "files."


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
