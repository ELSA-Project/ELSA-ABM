# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:06:18 2013

@author: earendil
"""
#from networkx import shortest_path
#import numpy as np
from simAirSpaceO import Net
import pickle
import networkx as nx
from scipy.spatial import Voronoi
import numpy as np

version='2.9.0'

def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))
def segments(p):
    return zip(p, p[1:] + [p[0]])

def prepare_network(paras_G, generate_new=False):
    ####################### Generate/load network #####################
    
#    type_of_net='D'                 #type of graph ('D'=Delaunay triangulation, 'T'=triangular lattice, "E"=Erdos-Renyi random graph)
#    N=30                            #order of the graph (in the case net='T' verify that the order is respected)        
#    #airports=[65,20]                #IDs of the nodes used as airports
#    airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
#    nairports=2                     #number of airports
#    pairs=[]#[(22,65)]              #available connections between airports
#    #[65,20]
#    #[65,22]
#    #[65,62]
#    min_dis=2                       #minimum number of nodes between two airpors
    
    #network_name='generic'
    
    generate_new=False
    
    #airport_dis=len(shortest_path(G,airports[0],airports[1]))-1  #this is the distance between the airports
    
    G=Net()
    if generate_new:
        G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'])
    else:
        if 1:
            fille='DEL_C_65_20'
            type_of_net='D'
            
        if 0:
            fille='test_graph_90_weighted_II'
            type_of_net='D' 
        
        if 0:
            fille='Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0_undirected_threshold'
            type_of_net='R' 
            
        _g=open(fille + '.pic','r')           
        H=pickle.load(_g)
        _g.close()
        
        G.import_from(H)
        if paras_G['airports']!=[]:
            G.fix_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
        else:
            G.generate_airports(paras_G['nairports'],paras_G['min_dis'])
        G.build_H()
    
    ####################### Capacities/weights #####################
    
#    generate_weights=True
#    typ_weights='coords'
#    sigma=0.01
#    generate_capacities=True
#    typ_capacities='manual'
#    C=5                             #sector capacity
    
    #file_capacities='capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'
    
    if paras_G['compute_areas']:        
        vor = Voronoi(np.array([G.node[n]['coord'] for n in G.nodes()]))
        for i,p in enumerate(vor.point_region):
            r = vor.regions[p]
            if len(r)>2:
                G.node[i]['area']=area([vor.vertices[n] for n in  r + [r[0]] if -1<=vor.vertices[n][0]<=1 and -1<=vor.vertices[n][1]<=1 and n!=-1])
    
    if paras_G['generate_capacities']:
        G.generate_capacities(typ=paras_G['typ_capacities'], C=paras_G['C'], par=paras_G['suppl_par_capacity'], file_capacities=paras_G['file_capacities'])
        G.typ_capacities=paras_G['typ_capacities']
    else:
        G.typ_capacities='constant'
        
    if paras_G['generate_weights']:
        G.generate_weights(typ=paras_G['typ_weights'], par=[1.,paras_G['sigma']])
        G.typ_weights=paras_G['typ_weights']
    else:
        G.typ_weights='gauss'
        
    for a in G.airports:
        #G.node[a]['capacity_airport']=paras_G['C_airport']
        G.node[a]['capacity_airport']=100000
        G.node[a]['capacity']=10000
    
    ####################### Preprocess stuff ####################
    
    G.Nfp=10   ############################## ATTENTIONNNNNNN ###################
    
    G.initialize_load()#2*(nx.diameter(G)+G.Nfp))
    G.compute_shortest_paths(G.Nfp)
    
    print 'Number of nodes:', (len(G.nodes()))
    
    ##################### Name ###########################
    
    long_name=type_of_net + '_N' + str(len(G.nodes()))
    
    if paras_G['airports']!=[]:
       long_name+='_airports' +  str(paras_G['airports'][0]) + '_' + str(paras_G['airports'][1])
    if paras_G['pairs']!=[] and len(paras_G['airports'])==2:
        long_name+='_direction_' + str(paras_G['pairs'][0][0]) + '_' + str(paras_G['pairs'][0][1])
    long_name+='_cap_' + G.typ_capacities
    
    if G.typ_capacities!='manual':
        long_name+='_C' + str(paras_G['C'])
    long_name+='_w_' + G.typ_weights
    
    if G.typ_weights=='gauss':
        long_name+='sig' + str(paras_G['sigma'])
    long_name+='_Nfp' + str(G.Nfp)
    
    ##################### Manual name #################
    if paras_G['name']!='':
        name=paras_G['name']
    else:
        name=long_name
        
    G.name=name
    G.comments={'long name':long_name, 'made with version':version}
    G.basic_statistics(rep=name  + '_')
    
    f=open(name + '.pic','w')
    pickle.dump(G, f)
    f.close()
    
    print 'Done.'
    
    return G
    
if  __name__=='__main__':
    type_of_net='D'                 #type of graph ('D'=Delaunay triangulation, 'T'=triangular lattice, "E"=Erdos-Renyi random graph)
    N=30                            #order of the graph (in the case net='T' verify that the order is respected)        
    #airports=[65,20]                #IDs of the nodes used as airports
    #airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
    #airports=[65,22,10,45, 30, 16]
    airports=[65,20]
    nairports=len(airports)                  #number of airports
    pairs=[]#[(22,65)]              #available connections between airports
    #[65,20]
    #[65,22]
    #[65,62]
    min_dis=2                       #minimum number of nodes between two airpors
    
    
    generate_weights=True
    typ_weights='coords'
    sigma=0.01
    generate_capacities=True
    typ_capacities='areas'
    C=5                             #sector capacity
    C_airport=10
    compute_areas=True
    #suppl_par_capacity=[0.3]
    suppl_par_capacity=['sqrt']
    
    
    file_capacities='capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'
    
   
    #name='LF_R_NSE'
    
    paras_G={k:v for k,v in vars().items() if k[:1]!='_' and k!='version' and k!='prepare_network'}
    
    if 0:
        name='DEL26_A_65_20'
        G=prepare_network(paras_G)
    else:
        pass


