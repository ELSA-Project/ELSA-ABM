#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:38:09 2012

@author: earendil
"""

from iter_simO import iter_sim, yes, build_path_average
from ABMvars import paras
from simulationO import Simulation, post_process_queue, extract_aggregate_values_on_queue, extract_aggregate_values_on_network
import numpy as np

def build_path(paras, vers = '2.9', **kwargs):
    rep, name = build_path_average(paras, vers = vers, **kwargs)
    return rep, name + '_shocks'
            
def do_shocks((paras, G, i)):
    results ={}

    sim=Simulation(paras, G=G.copy(), verbose=False)
    sim.make_simu(storymode=False)
    sim.queue=post_process_queue(sim.queue)

    G = sim.G

    results_queue=extract_aggregate_values_on_queue(sim.queue, paras['par'])
    queue_res = [f for f in sim.queue if G.G_nav.has_node(f.source) and G.G_nav.has_node(f.destination) and G.G_nav.node[f.source]['sec']!=paras['STS'] and G.G_nav.node[f.destination]['sec']!=paras['STS']]
    print 'len(queue_res):', len(queue_res), 'len(sim.queue)', len(sim.queue)
    results_queue_res=extract_aggregate_values_on_queue(queue_res, paras['par'])
    #print 'Satisfaction', results_queue['satisfaction']
    results_G=extract_aggregate_values_on_network(sim.G)

    # results_G_details={'load_details':{}, 'load_norm_details':{}}
    # for n in G:

    #     results_G_details['load_details'][n] = np.mean(G.node[n]['load'])
    #     results_G_details['load_norm_details'][n] = np.mean(G.node[n]['load'])/float(G.node[n]['capacity'])

    for met in results_G:
        results[met]=results_G[met]
    #for met in results_G_details:
    #    results[met]=results_G_details[met]

    for met in results_queue_res:
        results_queue[met + '_res'] = results_queue_res[met]

    for met in results_queue:
        results[met]={tuple(p):[] for p in paras['par']}
        for p in paras['par']:
            results[met][tuple(p)]=results_queue[met][tuple(p)]

    del sim
    return results


if __name__=='__main__':
    if yes('Ready?') or 1:
        results=iter_sim(paras, do = do_shocks, build_pat = build_path)
    
    print 'Done.'
    
