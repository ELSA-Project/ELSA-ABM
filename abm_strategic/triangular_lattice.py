#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:51:59 2013

@author: earendil
"""
from math import *
import networkx as nx
import matplotlib.pyplot as plt

def build_triangular(N):
    """
    build a triangular lattice in a rectangle with N nodes along the abscissa (so 4*N**2 in total)
    
    argument:
        N: number of nodes along the abscissa
    output:
        G: Networkx Graph with nodes andd edges
    """
    eps=10e-6
    G=nx.Graph()   
    a=1./float(N+0.5) - eps
    n=0
    j=0
    while j*sqrt(3.)*a <= 1.:
        i=0
        while i*a <= 1.:
            G.add_node(n, coords=(i*a, j*sqrt(3.)*a)) #LUCA: node capacity added.
            n+=1
            if i*a + a/2 < 1. and  j*sqrt(3.)*a + (sqrt(3.)/2.)*a < 1.:
                G.add_node(n, coords=(i*a + a/2., j*sqrt(3.)*a + (sqrt(3.)/2.)*a)) #LUCA: node capacity added.
                n+=1
            i+=1
        j+=1
            
    for n in G.nodes():
        for m in G.nodes():
            if n!=m and abs(sqrt((G.node[n]['coords'][0] - G.node[m]['coords'][0])**2\
            + (G.node[n]['coords'][1]- G.node[m]['coords'][1])**2) - a) <eps:
                G.add_edge(n,m)
    print(len(G.nodes()))
    
    return G
    
if __name__=='__main__':
    G=build_triangular(8.01783725737,5)
    
    plt.plot([G.node[n]['coords'][0] for n in G.nodes()], [G.node[n]['coords'][1] for n in G.nodes()], 'ro')
    for e in G.edges():
        plt.plot([G.node[e[0]]['coords'][0], G.node[e[1]]['coords'][0]], [G.node[e[0]]['coords'][1], G.node[e[1]]['coords'][1]], 'r-')
    plt.show()
