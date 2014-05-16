#!/usr/bin/env python

# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'../Distance')

import matplotlib.pyplot as plt
from tools_airports import extract_flows_from_data, get_paras
import numpy as np
import pickle


with open('LF29_R_FL350_DA0.pic', 'r') as f:
	G=pickle.load(f).G_nav

paras=get_paras()
paras['zone']=G.country
paras['airac']=G.airac
paras['n_days']=1
paras['type_zone']='EXT'
paras['filtre']='Strong'
paras['mode']='navpoints'
paras['cut_alt']=240.
paras['both']=False

flows, times=extract_flows_from_data(paras, [G.node[n]['name'] for n in G.nodes()])

print 'Number of flights:', sum(flows.values())

times=[item for sublist in times.values() for item in sublist]

times=(np.array(times) -  min(times))/60.
times=times[times<24*60.]

with open('times_2010_5_6.pic', 'w') as f:
	pickle.dump(times, f)

plt.hist(times, bins=50)

plt.show()

