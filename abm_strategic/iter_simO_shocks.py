#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:38:09 2012

@author: earendil

===========================================================================
This file contains a alternative to the do_standard procedure of iter_simO
 in order to study the impact of shocks on sectors.
===========================================================================
UNTESTED.
"""

import numpy as np

from iter_simO import iter_sim, yes, build_path_average
from simulationO import Simulation, post_process_queue, extract_aggregate_values_on_queue, extract_aggregate_values_on_network
from utilities import read_paras_iter

version = '2.9.1'

def build_path(paras, vers='2.9', **kwargs):
    rep, name = build_path_average(paras, vers = vers, **kwargs)
    return rep, name + '_shocks'
            
def do_shocks((paras, G)):
    """
    Make the simulation and extract aggregate values, keeping separate the results
    for flights which have not been cancelled. 

    Parameters
    ----------
    paras : dictionary
        containing the parameters for the simulations.
    G : hybrid network

    Returns
    -------
    results : dictionary
        gathers results from extract_aggregate_values_on_queue and 
        extract_aggregate_values_on_network.
    
    Notes
    -----
    Changed in 2.9.1: changed signature, removed integer i.

    """

    results ={}

    sim = Simulation(paras, G=G.copy(), verbose=False)
    sim.make_simu(storymode=False)
    sim.queue = post_process_queue(sim.queue)

    G = sim.G

    results_queue = extract_aggregate_values_on_queue(sim.queue, paras['par'])
    queue_res = [f for f in sim.queue if G.G_nav.has_node(f.source) and G.G_nav.has_node(f.destination) and G.G_nav.node[f.source]['sec']!=paras['STS'] and G.G_nav.node[f.destination]['sec']!=paras['STS']]
    results_queue_res = extract_aggregate_values_on_queue(queue_res, paras['par'])
    results_G = extract_aggregate_values_on_network(sim.G)

    for met in results_G:
        results[met]=results_G[met]

    for met in results_queue_res:
        results_queue[met + '_res'] = results_queue_res[met]

    for met in results_queue:
        results[met]={tuple(p):[] for p in paras['par']}
        for p in paras['par']:
            results[met][tuple(p)]=results_queue[met][tuple(p)]

    del sim
    return results

if __name__=='__main__':
    if yes('Ready?'):
        results = iter_sim(read_paras_iter(), do=do_shocks, build_pat=build_path)
    
    print 'Done.'
    
