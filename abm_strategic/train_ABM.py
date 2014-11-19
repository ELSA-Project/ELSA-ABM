#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is used to calibrate, or train, the ABM. It sweeps some parameter and 
computes the distance between distributions od degree, strength, betweenness, 
and weights. The default distance is the kolmogorov distance. The sweep is a 
brute force sweep and returns the best value of the parameter for each distribution.
"""

import sys
sys.path.insert(1,'../Distance')
#from analyse_network import network_and_trajectories_of_real_traffic, network_and_trajectories_of_simulated_traffic
from sklearn import metrics
from utilitiesO import compare_networks
#from scipy.integrate import quad
from simulationO import Simulation, post_process_queue
import ABMvars
from iter_simO import build_path_average
from performance_plots import build_path
from time import time, gmtime, strftime
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from string import split
from general_tools import loading

version = '2.9.0'
main_version=split(version,'.')[0] + '.' + split(version,'.')[1]

loc={'ur':1, 'ul':2, 'll':3, 'lr':4, 'r':5, 'cl':6, 'cr':7, 'lc':8, 'uc':9, 'c':10}


def loop(a, level, parass, results, thing_to_do=None, **args):
    """
    """
    if level==[]:
    	results = thing_to_do(**args)
        return results
    else:
        assert level[0] in a.keys()
        for i in a[level[0]]:
            print level[0], '=', i
            parass.update(level[0], i)
            results[i] = loop(a, level[1:], parass, {}, thing_to_do=thing_to_do, **args)

    return results

def average_sim(paras=None, G=None, save=1, suffix = '', stat_distance = 'common_area'):
	"""
	New in 2.6: makes a certain number of iterations (given in paras) and extract the averaged mettrics.
	Change in 2.7: parallelized.
	Changed in 2.9.1: added force 
	"""
	def do((paras, G,i)):
		rlts={} 
		sim=Simulation(paras, G=G.copy(), verbose=False)
		sim.make_simu(storymode=False)
		sim.queue=post_process_queue(sim.queue)

		rlts = compare_networks(sim, stat_distance = stat_distance)

		del sim
		return rlts

	rep, name = build_path_average(paras, Gname=paras['G'].name)

	name = name[:-4] + suffix + '.pic' #_comparison_distributions.pic'

	if paras['force'] or not os.path.exists(rep + name):
		inputs = [(paras, G, i) for i in range(paras['n_iter'])]
		start_time=time()
		if paras['parallel']:
			print 'Doing iterations',
			results_list = parmap(do, inputs)
		else:
			results_list=[]
			for i,a in enumerate(inputs):
				sys.stdout.write('\r' + 'Doing simulations...' + str(int(100*(i+1)/float(paras['n_iter']))) + '%')
				sys.stdout.flush() 
				results_list.append(do(a))

		print '... done in', time()-start_time, 's'
		results={}
		for met in results_list[0].keys():
			#print met, results_list[0][met], type(results_list[0][met]), type(np.float64(1.0))
			if type(results_list[0][met])==type(np.float64(1.0)) or type(results_list[0][met])==type(1.0):
				results[met]={'avg':np.mean([v[met] for v in results_list]), 'std':np.std([v[met] for v in results_list])}
			elif type(results_list[0][met])==type({}):
				results[met]={tuple(p):[] for p in results_list[0][met].keys()}
				for company in results_list[0][met].keys():
					results[met][company]={'avg':np.mean([v[met][company] for v in results_list]), 'std':np.std([v[met][company] for v in results_list])}

		if save>0:
			#rep, name=build_path_average(paras, Gname=G.name)
			os.system('mkdir -p ' + rep)
			with open(rep + name,'w') as f:
				pickle.dump(results, f)
				#print 'Saving in', rep + name
	else:
		print 'Skipped this value because the file already exists and parameter force is deactivated.'
		print 'I load it from disk.'
		with open(rep + name,'r') as f:
			results = pickle.load(f)

	return results

def find_best(results, target):
	x = sorted(results.keys())
	y = [results[xx][target]['avg'] for xx in x]

	return x[np.argmax(y)]

@loading
def compute_distance(paras, rep, G, stat_distance = 'kolmogorov', force = False):
	results = loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']}, paras['paras_to_loop'],\
 		paras, {}, thing_to_do=average_sim, paras=paras, G=G, suffix = stat_distance, stat_distance = stat_distance)
	return results

def calibrate_ABM(paras):
	if paras['fixnetwork']:
		G=paras['G']        
	else:
		G=None

	rep = build_path(paras, vers = main_version)
	results = compute_distance(paras, rep, G, stat_distance = 'kolmogorov', save =True, path = rep + '/results_calibration.pic')
	
	print
	print
	print 'The best parameter for degree is', paras['paras_to_loop'][0], '=', find_best(results, 'deg')
	y = [results[xx]['deg']['avg'] for xx in sorted(results.keys())]
	print 'The min/max are', min(y), '/', max(y)
	print 'The best parameter for strength is', paras['paras_to_loop'][0], '=', find_best(results, 'str')
	y = [results[xx]['str']['avg'] for xx in sorted(results.keys())]
	print 'The min/max are', min(y), '/', max(y)
	print 'The best parameter for weight is', paras['paras_to_loop'][0], '=', find_best(results, 'wei')
	y = [results[xx]['wei']['avg'] for xx in sorted(results.keys())]
	print 'The min/max are', min(y), '/', max(y)
	print 'The best parameter for length is', paras['paras_to_loop'][0], '=', find_best(results, 'len')
	y = [results[xx]['len']['avg'] for xx in sorted(results.keys())]
	print 'The min/max are', min(y), '/', max(y)

	plot_target(results, rep, paras, target = 'deg')
	plot_target(results, rep, paras, target = 'str')
	plot_target(results, rep, paras, target = 'wei')
	plot_target(results, rep, paras, target = 'len')
	plot_all_targets(results, rep, paras, loc = loc['ur'])
	plt.show()

	# for k, v in results.items():
	# 	paras[paras_to_loop]

def plot_all_targets(results, rep, paras, targets = ['deg', 'str', 'len', 'wei'], loc = 1):
	labely = 'Similarity'
	labelx = paras['paras_to_loop']

	plt.figure()
	plt.title('Similarity between real and simulated dists')
	leg = []
	for target in targets:
		x = sorted(results.keys())
		y = [results[xx][target]['avg'] for xx in x]
		ey = [results[xx][target]['std'] for xx in x]
		plt.errorbar(x, y, ey, fmt='o--')
		leg.append(target)

	if loc!=0:
		plt.legend(leg, loc=loc) 
	
	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.savefig(rep + '/calibration_all_targets.png')

def plot_target(results, rep, paras, target = 'deg'):
	x = sorted(results.keys())
	y = [results[xx][target]['avg'] for xx in x]
	ey = [results[xx][target]['std'] for xx in x]

	labely = 'Similarity'
	labelx = paras['paras_to_loop']

	plt.figure()
	plt.title('Similarity between real and simulated dist. of ' + target)
	plt.errorbar(x, y, ey, fmt='o--', color='r')
	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.savefig(rep + '/calibration_' + target + '.png')


if __name__ == '__main__':
	paras = ABMvars.paras

	# a = range(6)
	# f_p = [0., 1., 0., 0., 0.]

	# f = make_function(a, f_p)
	# ff = norm_function(a, f)

	# b = [0.5*i for i in range(11)]
	# f_q = [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]
	# print len(b)
	# print len(f_q)

	# g = make_function(b, f_q)
	# gg = norm_function(b, g)

	# f_c = make_common_function(ff, gg)

	# A = compare_distribution(a, f_p, b, f_q)

	calibrate_ABM(paras)






