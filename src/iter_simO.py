#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:38:09 2012

@author: earendil

===========================================================================
This is the main interface to the model. The main functions are 
 - do_standard, which makes a single iteration of the model, 
 - average_sim, which makes several iterations of the model with the same
parameters, 
 - iter_sim, which makes iterations of average_sim over several values of 
 parameters. 
===========================================================================
"""

from simulationO import Simulation, build_path as build_path_single, post_process_queue, extract_aggregate_values_on_queue, extract_aggregate_values_on_network
import pickle
#import ABMvars
import os
from string import split
import numpy as np
import sys

from multiprocessing import Process, Pipe
from itertools import izip
from time import time, gmtime, strftime

from general_tools import yes

version='2.9.3'
main_version=split(version,'.')[0] + '.' + split(version,'.')[1]

# This is for parallel computation.
def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]


#--------------------------------------------------------#

def header(paras, paras_to_display=['departure_times','par','nA', 'ACtot', 'density','Delta_t','ACsperwave']):
    """
    Used just to show the main parameters in the console.
    """
    if paras_to_display==[]:
        paras_to_display=paras.keys()
        
    for p in paras['paras_to_loop']:
        try:
            paras_to_display.remove(p)
        except:
            pass
        
    first_line='------------------------ Program iter_sim version ' + version + ' ------------------------'
    l=len(first_line)
    trait='-'*l
    
    head=trait + '\n' + first_line + '\n' + trait + '\n'
    head+='Network: ' + paras['G'].name + ' ; '
    head+='Paras to loop on: ' + str(paras['paras_to_loop']) + ' ;\n'
    line=''
    for k in paras_to_display:
        try:
            v=paras[k]
            pouic=k + ': ' + str(v) +  ' ; '
            if len(line) < l - len(pouic):
                line = line + pouic
            else:
                head = head + line + '\n' 
                line=pouic
        except:
            pass
    head +=line + '\n' + trait +'\n'
    head += 'Started on ' + strftime("%a, %d %b %Y %H:%M:%S", gmtime()) + '\n'
    head += trait + '\n'
    return head
    
def build_path_average(paras, vers=main_version, in_title=['tau', 'par', 'ACtot', 'nA'], Gname=None):
    if Gname==None:
        Gname=paras['G'].name
    
    rep='../results/Sim_v' + vers + '_' + Gname
    
    return rep, '/' + build_path_single(paras,vers=vers, only_name = True) + '_iter' + str(paras['n_iter']) + '.pic'
    
def loop(a, level, parass, thing_to_do=None, **args):
    """
    Generic recursive function to make several levels of iterations. 
    New in 2.6: Makes an arbitrary number of loops
    a: dictionnary, with keys as parameters to loop on and values as the values on which to loop.
    level: list of parameters on which to loop. The first one is the most outer loop, the last one is the most inner loop.
    """
    if level==[]:
        thing_to_do(**args)#(paras, G)
    else:
        assert level[0] in a.keys()
        for i in a[level[0]]:
            print level[0], '=', i
            parass.update(level[0],i)
            loop(a, level[1:], parass, thing_to_do=thing_to_do, **args)
   
def do_standard((paras, G, i)):
    """
    Make the simulation and extract aggregate values.
    New in 2.9.2: extracted from average_sim
    """
    results={} 
    sim=Simulation(paras, G=G.copy(), verbose=False)
    sim.make_simu(storymode=False)
    sim.queue=post_process_queue(sim.queue)
    
    results_queue=extract_aggregate_values_on_queue(sim.queue, paras['par'])
    results_G=extract_aggregate_values_on_network(sim.G)
    
    for met in results_G:
        results[met]=results_G[met]
            
    for met in results_queue:
        results[met]={tuple(p):[] for p in paras['par']}
        for p in paras['par']:
            results[met][tuple(p)]=results_queue[met][tuple(p)]

    del sim
    return results
 
def average_sim(paras=None, G=None, save=1, do = do_standard, build_pat = build_path_average):#, mets=['satisfaction', 'regulated_F', 'regulated_FPs']):
    """
    Average some simulations which have the same 
    New in 2.6: makes a certain number of iterations (given in paras) and extract the averaged mettrics.
    Change in 2.7: parallelized.
    Changed in 2.9.1: added force.
    Changed in 2.9.2: added do and build_pat kwargs. 
    """

    rep, name=build_pat(paras, Gname=paras['G'].name)

    if paras['force'] or not os.path.exists(rep + name):  
        inputs = [(paras, G, i) for i in range(paras['n_iter'])]
        start_time=time()
        if paras['parallel']:
            print 'Doing iterations',
            results_list = parmap(do, inputs)
        else:
            results_list=[]
            for i,a in enumerate(inputs):
                #sys.stdout.write('\r' + 'Doing simulations...' + str(int(100*(i+1)/float(paras['n_iter']))) + '%')
                #sys.stdout.flush() 
                results_list.append(do(a))
            
            
        print '... done in', time()-start_time, 's'
        
        results={}
        for met in results_list[0].keys():
            if type(results_list[0][met])==type(np.float64(1.0)):
                results[met]={'avg':np.mean([v[met] for v in results_list]), 'std':np.std([v[met] for v in results_list])}
            elif type(results_list[0][met])==type({}):
                results[met]={tuple(p):[] for p in results_list[0][met].keys()}
                for company in results_list[0][met].keys():
                    results[met][company]={'avg':np.mean([v[met][company] for v in results_list]), 'std':np.std([v[met][company] for v in results_list])}
                    
        if save>0:
            rep, name=build_pat(paras, Gname=G.name)
            os.system('mkdir -p ' + rep)
            with open(rep + name,'w') as f:
                pickle.dump(results, f)
    else:
        print 'Skipped this value because the file already exists and parameter force is deactivated.'

def iter_sim(paras, save=1, do = do_standard, build_pat = build_path_average):#, make_plots=True):#la variabile test_airports è stata inserita al solo scopo di testare le rejections
    """
    Used to loop and average the simulation. G can be passed in paras if fixnetwork is True.
    save can be 0 (no save), 1 (save agregated values) or 2 (save all queues).
    Changed in 2.9.2: added do and build_pat kwargs.
    """

    # Used for debugging.
    # if 0:
    #     f=open('state.pic','w')
    #     pickle.dump(getstate(),f)
    #     f.close()
    # else:
    #     f=open('state.pic','r')
    #     setstate(pickle.load(f))
    #     f.close()

    print header(paras)
    
    if paras['fixnetwork']:
        G=paras['G']        
    else:
        G=None
        
    loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']}, paras['paras_to_loop'], \
        paras, thing_to_do=average_sim, paras=paras, G=G, do = do, build_pat = build_pat)
    
if __name__=='__main__':

    #if yes('Ready?'):
    #    results=iter_sim(ABMvars.paras)
    
    print 'Done.'
    
