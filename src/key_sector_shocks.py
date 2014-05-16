#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ABMvars import paras
#from performance_plots import get_results
from iter_simO import build_path_average
from iter_simO_shocks import build_path
import os
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from general_tools import flip_polygon
import pickle

def linear_mapping(sat, min_sat, max_sat):
	return ((max_sat - sat)/(max_sat - min_sat), 0., 0.)

def draw_critical_sectors(G, satisfactions,  rep='.', save=True, name='critical_sectors', show=True, flip_axes=False, dpi = 100):
	
	polygons = G.polygons

	#print satisfactions

	if flip_axes:
		# if G:
		# 	for n in G.nodes():
		# 		G.node[n]['coord']=(G.node[n]['coord'][1], G.node[n]['coord'][0])
		polygons={k:flip_polygon(pol) for k,pol in polygons.items()}

	print  'Drawing critical sectors map...'
	fig = plt.figure()
	ax = fig.add_subplot(111)

	min_sat = min(satisfactions.values())
	max_sat = max(satisfactions.values())

	print 'max_sat:', max_sat
	print 'min_sat:', min_sat
	print 'min_sat/max_sat:', min_sat/max_sat
	for n,pol in polygons.items():
		color = linear_mapping(satisfactions[n], min_sat, max_sat)
		patch = PolygonPatch(pol, alpha=0.5, zorder=2, color = color)
		ax.add_patch(patch) 



	plt.plot([0.6475], [43.4922222222222], 'bs', ms = 0.01)


	if save:
		plt.savefig(rep + '/' + name +'.png', dpi = dpi)
		print 'Figure saved in', rep + '/' + name +'.png'
	if show:
		plt.show()

	return fig

def get_results(paras, vers='2.9'):
    #results, ={}, {}
    results = loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']}, paras['paras_to_loop'], paras, {}, vers=vers)
    return results

def loop(a, level, parass, results, vers='2.9'):
    """
    New in 2.6: Makes an arbitrary number of loops
    a: dictionnary, with keys as parameters to loop on and values as the values on which to loop.
    level: list of parameters on which to loop. The first one is the most outer loop, the last one is the most inner loop.
    """
    if level==[]:
        rep, name= build_path(parass, vers=vers)
        with open(rep + name, 'r') as f:
            results=pickle.load(f)

        return results
    else:
        assert level[0] in a.keys()
        for i in a[level[0]]:
            # print level[0], '=', i
            parass.update(level[0],i)
            results[i]=loop(a, level[1:], parass, {}, vers=vers)
            
    return results

if __name__=='__main__':
	G = paras['G']
	results = get_results(paras, vers='2.9')

	#print results[0]
	rep, name=build_path(paras, Gname = G.name)
	rep += '/critical_sectors'
	os.system('mkdir -p ' + rep)

	satisfactions = {n:v['satisfaction'][(1., 0., 1000.)]['avg'] for n,v in results.items()}
	satisfactions_res = {n:v['satisfaction_res'][(1., 0., 1000.)]['avg'] for n,v in results.items()}

	draw_critical_sectors(G, satisfactions, rep = rep, flip_axes = True, name = 'critical_sectors', show = False)
	draw_critical_sectors(G, satisfactions_res, rep = rep, flip_axes = True, name = 'critical_sectors_res')