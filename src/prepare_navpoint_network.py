#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 07:01:33 2013

@author: earendil
"""
import sys
sys.path.insert(1,'../Distance')
from simAirSpaceO import Net, NavpointNet
import pickle
import networkx as nx
from scipy.spatial import Voronoi
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
from random import choice, seed, sample
from general_tools import draw_network_and_patches, silence, counter, delay, date_st, make_union_interval
from utilities import restrict_to_connected_components
from string import split

# A VIRER
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt

#Distance
from tools_airports import get_paras, extract_flows_from_data#, get_flights

#see=1200
#see_ = 15 #==> the one I used for new, old_good, etc.. 
#see=2
#seed(see_)

version='2.9.5'

_colors = ('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')

def show_polygons(polygons,nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for pol in polygons.values():
        patch = PolygonPatch(pol,alpha=0.5, zorder=2)
        ax.add_patch(patch) 
    plt.plot([n[0] for n in nodes], [n[1] for n in nodes],'ro')
    #plt.show()
    
def show_everything(polygons,G,save=True,name='network_navpoints', show=True):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for pol in polygons.values():
    #     patch = PolygonPatch(pol,alpha=0.5, zorder=2)
    #     ax.add_patch(patch) 
    # ax.scatter([G.node[n]['coord'][0] for n in G.nodes()], [G.node[n]['coord'][1] for n in G.nodes()],c='r', marker='s')
    # ax.scatter([G.G_nav.node[n]['coord'][0] for n in G.G_nav.nodes()], [G.G_nav.node[n]['coord'][1] for n in G.G_nav.nodes()],c=[_colors[G.G_nav.node[n]['sec']%len(_colors)]\
    #    for n in G.G_nav.nodes()],marker='o',zorder=6,s=100)
       
    # for e in G.G_nav.edges():
    #     plt.plot([G.G_nav.node[e[0]]['coord'][0],G.G_nav.node[e[1]]['coord'][0]],[G.G_nav.node[e[0]]['coord'][1],G.G_nav.node[e[1]]['coord'][1]],'k-',lw=0.2)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
        
       
    # if save:
    #     plt.savefig(name +'.png')

    draw_network_and_patches(G,G.G_nav,polygons, save=save, name=name,show=show)

def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])
    
def navpoints_at_borders(G,vor,lin_dens=10.):
    """
    Create navpoints at borders of a Voronoi tesselation.
    """
    navpoints=[]
    #borders=np.unique([(n1,n2) for for p in vor.point_region])
    borders={}
    for p in vor.point_region:
        r=vor.regions[p]
        for n1,n2 in [(r[i],r[i+1]) for i in range(len(r)-1)] + [(r[-1],r[0])]:
            borders[tuple(sorted((n1,n2)))]=0
        
    borders=borders.keys()
   # borders=np.unique([sorted(n1,n2) for p in vor.point_region for n1,n2 in [(vor.regions[p][i],vor.regions[p][i+1])\
   #     for i in range(len(vor.regions[p])-1)] + [(vor.regions[p][-1],vor.regions[p][0])]])
   #  j=0
   # for p in vor.point_region: # For each voronoi cell
   #     r=vor.regions[p]
   #     for n1, n2 in :
    for n1,n2 in borders:
        #for i in range(len(r)-1): # for each vertex
        #    n1, n2 = r[i], r[i+1]
            #if -1.<=vor.vertices[n2][0]<=1. and -1.<=vor.vertices[n2][1]<=1. and -1.<=vor.vertices[n1][0]<=1. and -1.<=vor.vertices[n1][1]<=1.:
            #if n1<n2:
                a=np.array(vor.vertices[n2] - vor.vertices[n1])
                l=np.linalg.norm(a) #length of border
                n_p=max(1,int(l*lin_dens)) # number of points to put on the border
                d=l/float(n_p+1)
                k=1
                change=False
                while d*k<l-10**(-4.):# put points until vertex is reached
                    x,y=list(vor.vertices[n1] + k*(d/l)*a +[10**(-5.), 10**(-5.)])
                    if -1.<x<1. and -1.<y<1.:
                        change=True
                        navpoints.append([x,y])
                    k+=1
                    #j+=1
                if not change:
                    print 'nothing happened:', n1,n2, vor.vertices[n1], vor.vertices[n2]
    return navpoints#j
   
def prepare_sectors_network(paras_G, mode = 'fake', generate_new_sectors=False, layer=0, no_airports=False):     
    G=Net()
    G.type='sec'
    if generate_new_sectors:
        G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'])
    else:
        if mode == 'fake':
            fille='DEL_C_6A'
            G.type_of_net='D'
            
        # if 0:
        #     fille='test_graph_90_weighted_II'
        #     G.type_of_net='D' 
        else:
            #fille='Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0_undirected_threshold'
            fille='../Modules/' + paras_G['country'] + '_sectors_network_one_layer_FL' + str(layer) + '_A334'
            G.type_of_net='R'  # For "Real"
            G.airac=paras_G['airac']
            G.country=paras_G['country']
            G.layer=layer #indicates the layer; '0' for full otherwise.

            
        _g=open(fille + '.pic','r')           
        H=pickle.load(_g)
        _g.close()
        
        
        
        if G.type_of_net=='R':  # Numberize the sectors (instead of initial names)
            G.idx_sectors={s:i for i,s in enumerate(H.nodes())}
            for n in H.nodes():
                G.add_node(G.idx_sectors[n], name=n, **H.node[n])
            for e in H.edges():
                G.add_edge(G.idx_sectors[e[0]], G.idx_sectors[e[1]])
        else:
            G.import_from(H)

        if not no_airports:
            if paras_G['airports']!=[]:
                G.fix_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
            else:
                G.generate_airports(paras_G['nairports'],paras_G['min_dis'])
            
        if G.type_of_net=='D':
            G, vor=compute_voronoi(G)
            G.vor=vor
        elif G.type_of_net=='R':
            with open('../Modules/sectors_network_one_layer_FL' + str(layer) + '_A334_polygons.pic', 'r') as f:
                polygons_names=pickle.load(f)
            G.polygons={}
            for name, shape in polygons_names.items():
                G.polygons[G.idx_sectors[name]]=shape
            # with open('../Modules/All_shapes_334.pic', 'r') as f:
            #     all_shapes=pickle.load(f)
            #     G.polygons={}
            #     for i in G.nodes():
            #         #print G.node[i]
            #         G.polygons[i]=all_shapes[G.node[i]['name']]['boundary'][0]
            # for n in G.nodes():
            #     G.node[n]['coord']=list(np.array(G.node[n]['coord'])/60.)
            G.global_shape=cascaded_union(G.polygons.values())
                
    return G
    
def compute_voronoi(G):
    polygons={}
    vor = Voronoi(np.array([G.node[n]['coord'] for n in G.nodes()]))
    for i,p in enumerate(vor.point_region):
        r = vor.regions[p]
        #coords=[[vor.vertices[n][1], vor.vertices[n][0]] for n in  r + [r[0]] if n!=-1]#-1<=vor.vertices[n][0]<=1 and -1<=vor.vertices[n][1]<=1]
        coords=[vor.vertices[n] for n in  r + [r[0]] if n!=-1]#-1<=vor.vertices[n][0]<=1 and -1<=vor.vertices[n][1]<=1]
        if len(coords)>2:
           
            # print coords
            G.node[i]['area']=area(coords)
            polygons[i]=Polygon(coords)
            try:
                assert abs(G.node[i]['area'] - polygons[i].area)<10**(-6.)
            except:
                raise Exception(i, G.node[i]['area'], polygons[i].area)
    
    eps=0.1
    minx,maxx,miny,maxy=min([n[0] for n in vor.vertices]) -eps, max([n[0] for n in vor.vertices]) +eps, min([n[1] for n in vor.vertices]) -eps, max([n[1] for n in vor.vertices]) +eps
    
    #big_square=Polygon([[minx,miny], [minx,maxy], [maxx,maxy], [maxx,miny],[minx,miny]], [[[-1,-1],[-1,1],[1,1],[1,-1],[-1,-1]]])
    square=Polygon([[-1,-1],[-1,1],[1,1],[1,-1],[-1,-1]])

    for n,pol in polygons.items():
        #polygons[n]=pol.difference(big_square)
        polygons[n]=pol.intersection(square)
        
    G.polygons=polygons
    return G, vor
        
def extract_airports_from_traffic(G, flows): #,paras_G):
    """
    New in 2.8: gives naiports "airports" among all points of entries and exits from data trajectories.
    Changed in 2.9: gives pairs as well. Gives all airports.
    Changed in 2.9.4: added paras_real.
    Chanded in 2.9.5: flos are given externally
    """
    # paras['zone']=G.country
    # paras['airac']=G.airac
    # paras['type_zone']='EXT'
    # paras['filtre']='Strong'
    # paras['mode']='navpoints'
    # paras['cut_alt']=240.
    # paras['both']=False
    # paras['n_days']=1

    all_airports=np.unique([k[0] for k in flows.keys()] + [k[1] for k in flows.keys()])
    all_pairs= flows.keys()
    # if paras_G['nairports']!=0:
    #     airports=sample(all_airports, paras_G['nairports'])
    # else:
    #     airports=all_airports

    #G.fix_airports([G.idx_sectors[airports, paras_G['min_dis'])

    return [G.G_nav.idx_navs[name] for name in all_airports], [(G.G_nav.idx_navs[p[0]], G.G_nav.idx_navs[p[1]]) for p in all_pairs]
    
def extract_weights_from_traffic(G, flights):
    """
    For each edge of G, computes the average time elapsed between two navpoints.
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: flights are given externally.
    """
    print 'Extracting weights from data...'

    #flights=get_flights(paras_real)
    weights={}
    pop={}
    for f in flights:
        r=f['route_m1t']
        for i in range(len(r)-1):
            if G.idx_navs.has_key(r[i][0]) and G.idx_navs.has_key(r[i+1][0]):
                p1=G.idx_navs[r[i][0]]
                p2=G.idx_navs[r[i+1][0]]
                if G.has_edge(p1,p2):
                    weights[(p1,p2)]= weights.get((p1, p2),0.) + delay(np.array(r[i+1][1]),starting_date=np.array(r[i][1]))/60.
                    pop[(p1,p2)]= pop.get((p1, p2),0.) + 1

    for k in weights.keys():
        weights[k]=weights[k]/float(pop[k])

    if len(weights.keys())<len(G.edges()):
        print 'Warning! Some edges do not have a weight!'
    return weights

def extract_capacity_from_traffic(G, flights, date = [2010, 5, 6, 0, 0, 0]):
    """
    New in 2.9.3: Extract the "capacity", the maximum number of flight per hour, based on the traffic.
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: fligths are given externally.
    """
    print 'Extracting the capacities...'

    #flights=get_flights(paras_real)
    weights={}
    pop={}
    loads = {n:[0 for i in range(48)] for n in G.nodes()}
    for f in flights:
        hours = {}
        r = f['route_m1t']
        for i in range(len(r)):
            if G.G_nav.idx_navs.has_key(r[i][0]):
                p1=G.G_nav.idx_navs[r[i][0]]
                if G.G_nav.has_node(p1):
                    s1=G.G_nav.node[p1]['sec']
                    hours[s1] = hours.get(s1,[]) + [int(float(delay(r[i][1], starting_date = date))/3600.)]

        for n,v in hours.items():
            hours[n] = list(set(v))

        for n,v in hours.items():
            for h in v:
                loads[n][h]+=1

    capacities = {}
    for n,v in loads.items():
        capacities[n] = max(v)
    
    return capacities

def extract_old_capacity_from_traffic(G, paras_real, date = [2010, 5, 6, 0, 0, 0]):
    """
    New in 2.9.4.
    """
    print 'Extracting the old capacities...'

    flights=get_flights(paras_real)
    weights={}
    pop={}

    intervals={G.node[n]['name']:[([2008,0,0,0,0,0],0)] for n in G.nodes()}

    for fk,f in flights.items():
        if f['route_m1t']!=[]:
            times=[n[1] for n in f['route_m1t']]
            # print 'List of navpoints:', [n[0] for n in f['route_m1t']]
            # print 'List of idx of navpoints:', [G.G_nav.idx_navs[n[0]] for n in f['route_m1t'] if n[0] in G.G_nav.idx_navs.keys()]
            # print 'List of idx of sectors:', [G.G_nav.node[G.G_nav.idx_navs[n[0]]]['sec'] for n in f['route_m1t'] if n[0] in G.G_nav.idx_navs.keys() and G.G_nav.has_node(G.G_nav.idx_navs[n[0]])]
            
            navs = [n[0] for n in f['route_m1t']]
            navs_idx = [G.G_nav.idx_navs[n] for n in navs if G.G_nav.idx_navs.has_key(n)]
            sects_idx = [G.G_nav.node[n]['sec'] for n in navs_idx if G.G_nav.has_node(n)]
            sects = [G.node[s]['name'] for s in sects_idx if G.has_node(s)]
            #sects=[G.G_nav.node[G.G_nav.idx_navs[n[0]]]['sec']] for n in f['route_m1t'] if n[0] in G.G_nav.idx_navs.keys()\
            # and G.G_nav.has_node(G.G_nav.idx_navs[n[0]])]
            #print 'List of sectors:', sects
            sectsU=sorted(set(sects), key=lambda c:sects.index(c))
            # print 'List of unique sectors:', sectsU
            # print
            if len(sectsU)!=0:
                for i in range(len(sectsU)-1):
                    intervals[sectsU[i]].append((times[sects.index(sectsU[i])], times[sects.index(sectsU[i+1])])) 
                
                intervals[sectsU[-1]].append((times[sects.index(sectsU[-1])], times[-1]))
        
    capacities={}
    for n in intervals.keys():
        capacities[G.idx_sectors[n]]=max([p[1] for p in make_union_interval(intervals[n])])
    
    return capacities

def find_pairs(all_airports, all_pairs, nairports, G, remove_pairs_same_sector = False):
    """
    New in 2.9: find nairports for which each of them have at least a link with one of the other ones.
    The algorithm should never fail for nairport=2 (you can always find a pair) and of course 
    for naiports=len(all_airports).
    New in 2.9.3: Possibility of removing the pairs which have their airports in the same sector.
    """
    if remove_pairs_same_sector:
        to_remove = []
        for p1, p2 in all_pairs:
            if G.G_nav.node[p1]['sec']==G.G_nav.node[p2]['sec']:
                to_remove.append((p1,p2))
        for p in to_remove:
            all_pairs.remove(p)

    black_list= []
    candidates=sample(all_airports, nairports)
    found=len(all_airports)==nairports
    pairs=[p for p in all_pairs if p[0] in candidates and p[1] in candidates]
    while not found and len(black_list)<(len(all_airports) - nairports):
        pairs=[p for p in all_pairs if p[0] in candidates and p[1] in candidates]
        selected=np.unique([p[0] for p in pairs] + [p[1] for p in pairs])
        found=True
        for candidate in candidates:
            if not candidate in selected:
                candidates.remove(candidate)
                black_list.append(candidate)
                found=False
        if not found:
            candidates += sample([a for a in all_airports if not a in black_list], nairports - len(candidates))
    if not found:
        raise Exception('Impossible to find pairs with that much airports. Try with less airports.')
    else:
        assert len(candidates)==nairports
        return candidates, pairs

def compute_sp_restricted(G, Nfp, silent=True):
    """
    New in 2.9.0: Computes the k shortest paths of navpoints restricted to shortest path of sectors.
    Changed in 2.9.1: save all Nfp shortest paths of navpoints for each paths of sectors (Nfp also).
    """
    G.G_nav.short={}
    G.short_nav={}
    for idx,(p0,p1) in enumerate(G.G_nav.pairs):
        counter(idx, len(G.G_nav.pairs), message='Computing shortest paths...')
        s0,s1=G.G_nav.node[p0]['sec'], G.G_nav.node[p1]['sec']
        G.short_nav[(p0,p1)]={}
        for idx_sp,sp in enumerate(G.short[(s0,s1)]): # Compute the network of navpoint restricted of each shortest paths.
            print 'shortest path in sectors:', sp
            H_nav=NavpointNet()
            with silence(silent):
                #H_nav.import_from(nx.subgraph(G.G_nav, [n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec'] in sp])) # restrict graph to possible paths of sectors.
                HH=nx.Graph()
                # Add every nodes in the sectors of the shortest paths.
                for n in G.G_nav.nodes(): 
                    if G.G_nav.node[n]['sec'] in sp:
                        HH.add_node(n, **G.G_nav.node[n])

                for e in G.G_nav.edges():
                    s0=G.G_nav.node[e[0]]['sec']
                    s1=G.G_nav.node[e[1]]['sec']
                    if s0!=s1 and s0 in sp and s1 in sp:
                        idxs_s0=np.where(np.array(sp)==s0)[0]
                        idxs_s1=np.where(np.array(sp)==s1)[0]
                        for idx_s0 in idxs_s0: # In case there is a repetition of s0 in sp.
                            for idx_s1 in idxs_s1:
                                if ((idx_s0<len(sp)-1 and sp[idx_s0+1]==s1) or (idx_s1<len(sp)-1 and sp[idx_s1+1]==s0)):
                                    HH.add_edge(*e, weight=G.G_nav[e[0]][e[1]]['weight'])
                    elif s0==s1 and s0 in sp: # if both nodes are in the same sector and this sector is in the shortest path, add the edge.
                        HH.add_edge(*e, weight=G.G_nav[e[0]][e[1]]['weight'])
                H_nav.import_from(HH)
                #print 'len(H_nav), len(HH)', len(H_nav), len(HH.nodes())

                if len(H_nav.nodes())!=0:
                    try:
                        #draw_network_and_patches(None, H_nav, G.polygons, show=True, flip_axes=True)#, trajectories=G.G_nav.short.values()[0])
                        for i in range(len(sp)-1):
                            found=False
                            ss1, ss2=sp[i], sp[i+1]
                            for n1, n2 in G.G_nav.edges():
                                if (G.G_nav.node[n1]['sec']==ss1 and G.G_nav.node[n2]['sec']==ss2) or (G.G_nav.node[n1]['sec']==ss2 and G.G_nav.node[n2]['sec']==ss1):
                                    found=True
                                    if ss1==0 and ss2==31:
                                        print n1, n2, G.G_nav.node[n1]['sec'], G.G_nav.node[n2]['sec'], H_nav.has_node(n1), H_nav.has_node(n1), H_nav.has_edge(n1,n2)
                            if found==False:
                                print 'Problem: sectors', ss1, 'and', ss2, 'are not adjacent in terms of navpoints.'
                            else:
                                print 'sectors', ss1 , 'and', ss2, 'are adjacent.'
                        for i in range(len(sp)-1):
                            found=False
                            ss1, ss2=sp[i], sp[i+1]
                            for n1, n2 in H_nav.edges():
                                if (G.G_nav.node[n1]['sec']==ss1 and G.G_nav.node[n2]['sec']==ss2) or (G.G_nav.node[n1]['sec']==ss2 and G.G_nav.node[n2]['sec']==ss1):
                                    found=True
                            if found==False:
                                print 'Problem: sectors', ss1, 'and', ss2, 'are not adjacent in terms of navpoints (H).'
                            else:
                                print 'sectors', ss1 , 'and', ss2, 'are adjacent (H).'

                        H_nav.fix_airports([p0,p1], 0.)
                        H_nav.build_H()
                        H_nav.compute_shortest_paths(Nfp, repetitions=False, use_sector_path=True, old=True)

                        for k, p in enumerate(H_nav.short[(p0,p1)]):
                            if G.convert_path(p)!=sp:
                                print 'Alert: discrepancy between theoretical path of sectors and final one!'
                                print 'Path number', k
                                print 'Theoretical one:', sp, G.weight_path(sp)
                                print 'Actual one:',  G.convert_path(p), G.weight_path(G.convert_path(p))        
                                raise                           

                        shorts=[p for p in  H_nav.short[(p0,p1)] if G.convert_path(p)==sp]
                        G.G_nav.short[(p0,p1)] = G.G_nav.short.get((p0,p1),[]) + shorts
                        G.short_nav[(p0,p1)][tuple(sp)] = shorts
                        # if shorts==[]:
                        #     print 'Alert, discrepancy between theoretical path of sectors and final one!'
                    except nx.NetworkXNoPath:
                        print 'No restricted shortest path between' ,p0, 'and', p1, 'but I carry on'
                        cc=nx.connected_components(H_nav)
                        print 'Composition of connected components (sectors):'
                        for c in cc:
                            print np.unique([G.G_nav.node[n]['sec'] for n in c])
                        #print 'Everybody is attached:', check_everybody_is_attached(H_nav)
                        #print H_nav.nodes()
                        #print H_nav.edges()
                        raise
                    except:
                        print 'Unexpected error:', sys.exc_info()[0]
                        raise 
                else:
                    print 'The subgraph was empty, I carry on.'
            print
        # if G.G_nav.short.has_key((p0,p1)):
        #     G.G_nav.short[(p0,p1)] = sorted(list(set([tuple(o) for o in G.G_nav.short[(p0,p1)]])), key= lambda p: G.G_nav.weight_path(p))[:Nfp]
    return G

def attach_two_sectors(s1, s2, G):
    navs_in_s1=[n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s1]
    navs_in_s2=[n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s2]
    pairs=[(n1, n2) for n1 in navs_in_s1 for n2 in navs_in_s2]
    distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
    n1_selected, n2_selected = pairs[np.argmin(distances)]
    G.G_nav.add_edge(n1,n2, weight=np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])))
    return G

def check_two_sectors_attached(s1, s2, G):
    found=False
    for n1, n2 in G.G_nav.edges():
        if (G.G_nav.node[n1]['sec']==s1 and G.G_nav.node[n2]['sec']==s2) or (G.G_nav.node[n1]['sec']==s2 and G.G_nav.node[n2]['sec']==s1):
            found=True
    return found

def check_everybody_is_attached(G, repair=False):
    problem=False
    for s1, s2 in G.edges():
        attached = check_two_sectors_attached(s1, s2, G)
        if not attached:
            print 'Problem: sectors', s1, 'and', s2, 'are not adjacent in terms of navpoints.'
            problem=True
            if repair:
                print "I'm repairing this."
                G=attach_two_sectors(s1, s2, G)
                assert check_two_sectors_attached(s1, s2, G)

    return G, problem

# def check_sector_is_connected(s, G):
#     nodes_in_s = [n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s]
#     H_nav_s = G.G_nav.subgraph(nodes_in_s)
#     cc = nx.connected_components(H)
#     if len(cc)>1:

def check_everybody_has_one_cc(G, repair=False):
    problem=False
    for s in G.nodes():
        nodes_in_s = [n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s]
        H_nav_s = G.G_nav.subgraph(nodes_in_s)
        cc = nx.connected_components(H_nav_s)
        if len(cc)>1:
            print 'Problem: sector', s, 'has more than one connected component (' + str(len(cc)) + ') exactly.'
            problem=True
            if repair:
                print "I'm fixing this."
                for j in range(len(cc)-1):
                    c1, c2 = cc[j], cc[j+1]
                    pairs=[(n1, n2) for n1 in c1 for n2 in c2]
                    distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                    n1_selected, n2_selected = pairs[np.argmin(distances)]
                    w=np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord']))
                    G.G_nav.add_edge(n1,n2, weight=w)
                    H_nav_s.add_edge(n1,n2, weight=w)
    return G, problem

def give_capacities_and_weights(G, mode, paras_G):
    if mode == 'real':
        flights_selected = G.flights_selected

    if paras_G['generate_capacities']:
        G.generate_capacities(typ=paras_G['typ_capacities'], C=paras_G['C'], par=paras_G['suppl_par_capacity'], file_capacities=paras_G['file_capacities'])
        G.typ_capacities=paras_G['typ_capacities']
    else:
        capacities = extract_capacity_from_traffic(G, flights_selected)
        G.fix_capacities(capacities)
        #G.typ_capacities='constant'

    # if 1:
    #     for a in G.airports:
    #         G.node[a]['capacity']=100000
    #         G.node[a]['capacity_airport'] = paras_G['C_airport']
    # else:
    #     print 'ALERT: NO INFINITE CAPACITY AT AIRPORTS.'


    if paras_G['generate_weights']:
        G.G_nav.generate_weights(typ='coords', par=5.)
    else:
        G.G_nav.fix_weights(extract_weights_from_traffic(G.G_nav, flights_selected), typ='data')
        for e in G.G_nav.edges():
            if not G.G_nav[e[0]][e[1]].has_key('weight'):
                print e

    G.generate_weights(typ=paras_G['typ_weights'], par=20.)
    G.build_H()
    G.G_nav.build_H()

    return G

def write_down_network(G):
    """
    New in 2.9.4: write down the atributes, nodes and edges of the sector network.
    Used in particular to convert 2.9 graph to 2.6.
    """
    with open(G.name + '_pieces.txt', 'w') as f:
        for n in G.nodes():
            print >>f, n
            print >>f, G.node[n]
        print >>f, ''
        for e1, e2 in G.edges():
            print >>f, (e1, e2)
            print >>f, G[e1][e2]
        print >>f, ''


    for f in G.flights_selected:
        p_nav = [G.G_nav.idx_navs[p[0]] for p in f['route_m1']]
        p_sec = G.convert_path(p_nav)

        entry_times=[0.]*(len(p_sec)+1)
        entry_times[0] = f['route_m1t'][0][1]

        #print f['route_m1t'][1]
        road = delay(f['route_m1t'][0][1])
        sec = G.G_nav.node[p_nav[0]]['sec']
        j=0
        for i in range(1,len(p_nav)):
            #print road, date_st(road)
            w = delay(f['route_m1t'][i][1]) - delay(f['route_m1t'][i-1][1])  #G.G_nav[p_nav[i-1]][p_nav[i]]['weight']
            #print f['route_m1t'][i][1], f['route_m1t'][i-1][1], w
            if G.G_nav.node[p_nav[i]]['sec']!=sec:
                j+=1
                entry_times[j] = date_st(road + w/2.)
                sec=G.G_nav.node[p_nav[i]]['sec']
            road += w
        entry_times[len(p_sec)] = date_st(road)


        f['sec_path'] = p_sec
        f['sec_path_t'] = entry_times

        # print f['route_m1t']
        # print p_sec
        # print entry_times

        #assert 1==0

    with open(G.name + '_pieces2.pic', 'w') as f:
        dic = {}
        for k, a in G.graph.items():
            dic[k] = a

        for k in G.__dict__.keys():
            if not k in ['graph', 'short_nav', 'node', 'H', 'G_nav', 'edge', 'adj']:
                dic[k] = G.__dict__[k]

        pickle.dump(dic, f)

def prepare_navpoint_network(paras_G, mode = 'fake', generate_new_navpoints=False, generate_new_sectors=False, layer=0, airports_from_data=False):

    ######### Prepare network of sectors ################
    G=prepare_sectors_network(paras_G, mode = mode, generate_new_sectors=generate_new_sectors, layer=layer, no_airports=airports_from_data)

    ############ Prepare nodes and edges of network fo navpoints ###########
    for n in G.nodes():
        G.node[n]['navs']=[]
        G.G_nav=NavpointNet()
        G.G_nav.type='nav'
        
    if generate_new_navpoints:
        #G.G_nav.navpoints_borders=navpoints_at_borders(G.G_nav, vor)#,lin_dens=20.)
        G.G_nav.build((paras_G['N_by_sectors']-1)*len(G.nodes()),paras_G['nairports'],paras_G['min_dis'],generation_of_airports=False, \
                sector_list=[G.node[n]['coord'] for n in G.nodes()], navpoints_borders=navpoints_at_borders(G.G_nav, G.vor), shortcut=0.01) #ATTENTION: min_dis is not the right one.
                
       # G.G_nav.show()
       # plt.show()
                
        #G.G_nav.show(stack=True)
        
    else:
        if mode == 'real':
            paras_real = paras_G['paras_real']
            G.paras_real = paras_real

            if 1:
                fille='Weak_EXTLF_2010-5-6_directed_navpoints_threshold'
                G.G_nav.type_of_net='R' 
                
            with open(fille + '.pic','r') as _g:           
                H=pickle.load(_g)
            
            G.G_nav.idx_navs={s:i for i,s in enumerate(H.nodes())}
            for n in H.nodes():
                G.G_nav.add_node(G.G_nav.idx_navs[n], name=n, **H.node[n])
            for e in H.edges():
                G.G_nav.add_edge(G.G_nav.idx_navs[e[0]], G.G_nav.idx_navs[e[1]])
            #G.global_shape = H.global_shape
            G.country = paras_G['country']
            G.airac = paras_G['airac']
                
            if G.G_nav.type_of_net=='R':  
                G.G_nav.country=paras_G['country']
                G.G_nav.airac=paras_G['country']
                for n in G.G_nav.nodes():
                    G.G_nav.node[n]['coord']=list(np.array(G.G_nav.node[n]['coord'])/60.) #Convert minutes in degrees
                for e in G.G_nav.edges(): # remove all edges which are geometrically outside of the global shape
                    if not G.global_shape.contains(LineString([G.G_nav.node[e[0]]['coord'], G.G_nav.node[e[1]]['coord']])):
                        G.G_nav.remove_edge(*e)
                G.G_nav=restrict_to_connected_components(G.G_nav)
            
    ######### Linking the two networks together ##########

    not_found=[]
    for nav in G.G_nav.nodes(): # Finding to which sector each navpoint belongs.
        found=False
        i=0
        while not found and i<len(G.nodes()):
            sec=G.nodes()[i]
            if G.polygons.has_key(sec) and G.polygons[sec].contains(Point(np.array(G.G_nav.node[nav]['coord']))):
                found=True
            i+=1
        if not found:
            not_found.append(nav)
            G.G_nav.remove_node(nav)
        else:
            G.G_nav.node[nav]['sec']=sec
            G.node[sec]['navs'].append(nav)
    print 'Navpoints for which I did not find a matching sector (I removed them from the network):', not_found
            

    ########### Choose the airports #############

    #Old_behavior: choose airport sectors first and find navpoints airports then.
    # New behavior: 
    old_behavior = False or mode=='real'

    if not old_behavior: # new behavior (choose first the airports for sectors.)
        if paras_G['airports']!=[]:
            G.fix_airports(paras_G['airports'], -10, pairs=paras_G['pairs'], C_airport = paras_G['C_airport'])
        else:
            G.generate_airports(paras_G['nairports'], -10, C_airports = paras_G['C_airport'])
        print 'G airports:', G.airports
        G.G_nav.infer_airports_from_sectors(G.airports, paras_G['min_dis'])
        
    else:
        if not airports_from_data:
            G.G_nav.generate_airports(paras_G['nairports'],paras_G['min_dis'])  
        else:
            flows = extract_flows_from_data(paras_real, [G.G_nav.node[n]['name'] for n in G.G_nav.nodes()])[0]
            all_airports, all_pairs=extract_airports_from_traffic(G, flows)
            if paras_G['nairports']==0:
                paras_G['nairports']=len(all_airports)
            airports, pairs = find_pairs(all_airports, all_pairs, paras_G['nairports'], G, remove_pairs_same_sector = True)
            print 'Selected at first', len(airports), 'airports and', len(pairs), 'pairs.'
            #airports_sects=sample(np.unique([G.G_nav.node[n]['sec'] for n in airports_nav]), paras_G['nairports'])
            G.G_nav.fix_airports(airports, paras_G['min_dis'], pairs=pairs, singletons=False)
        G.infer_airports_from_navpoints(paras_G['C_airport'], singletons=False) #singletons = mode=='real'


    # if generate_new_navpoints: # Choosing airports
    #     if mode == 'real' or old_behavior: # old behavior
    #         if paras_G['airports']!=[]:
    #             G.G_nav.fix_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
    #         else:
    #             G.G_nav.generate_airports(paras_G['nairports'],paras_G['min_dis'], C_airport = paras_G['C_airport'])
    #     if mode == 'fake' and not old_behavior:# new behavior:
    #         airports_nav=[choice([n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==sec]) for sec in G.airports]
    #         G.G_nav.fix_airports(airports_nav, paras_G['min_dis'])# add pairs !!!!!!!!!!!!!!!!1
    #     print 'G_nav airports:', G.G_nav.airports
    # else:
    #     if not airports_from_data:
    #         G.G_nav.generate_airports(paras_G['nairports'],paras_G['min_dis'])  
    #     else:
    #         all_airports, all_pairs=extract_airports_from_traffic(G)
    #         if paras_G['nairports']==0:
    #             paras_G['nairports']=len(all_airports)
    #         airports, pairs = find_pairs(all_airports, all_pairs, paras_G['nairports'], G, remove_pairs_same_sector = True)
    #         print 'Selected at first', len(airports), 'airports and', len(pairs), 'pairs.'
    #         #airports_sects=sample(np.unique([G.G_nav.node[n]['sec'] for n in airports_nav]), paras_G['nairports'])
    #         G.G_nav.fix_airports(airports, paras_G['min_dis'], pairs=pairs)

    # if mode=='real' or old_behavior: # old behavior
    #     G.infer_airports_from_navpoints(paras_G['C_airport'])
    # G.G_nav.pairs=[]
    # for p in G.pairs:
    #     found=False
    #     j=-1
    #     while not found:
    #         j+=1
    #         if G.node[p[0]]['navs'][j] in G.G_nav.airports:
    #             found=True
    #     p0=G.node[p[0]]['navs'][j]
    #     G.node[p[0]]['airport_nav']=p0
    #     found=False
    #     j=-1
    #     while not found:
    #         j+=1
    #         if G.node[p[1]]['navs'][j] in G.G_nav.airports:
    #             found=True
    #     p1=G.node[p[1]]['navs'][j]  
    #     G.node[p[1]]['airport_nav']=p1
    
    #     G.G_nav.pairs.append((p0,p1))
            
    #G.G_nav.clean_borders()
     
    #show_everything(G.polygons,G, show=True)  
    
    ############# Repair some stuff #############

    idx=max(G.G_nav.nodes())
    #change=True
    change=generate_new_navpoints
    
    # We check if edges of navpoints are crossing sectors other than the two sectors of the two nodes.
    while change:
        change=False
        for nav in G.G_nav.nodes():
            sec1=G.G_nav.node[nav]['sec']
            for neigh in G.G_nav.neighbors(nav):
                sec2=G.G_nav.node[neigh]['sec']
                if sec1<sec2 and not G[sec1].has_key(sec2): # If sec1 the first node is different from sec2 and they are not neighbors, we have a problem.
                    sects=[s for s in G.nodes() if s!=sec1 and s!=sec2 and \
                         LineString([G.G_nav.node[nav]['coord'], G.G_nav.node[neigh]['coord']]).intersects(G.polygons[s])] # List of sectors geometrically crossed by the edge
                    idxs=[]
                    for s in sects:
                        l_inter=LineString([G.G_nav.node[nav]['coord'], G.G_nav.node[neigh]['coord']]).intersection(G.polygons[s]) # piece of line in the sector s
                        if type(l_inter)==type(LineString([G.G_nav.node[nav]['coord'], G.G_nav.node[neigh]['coord']])): # Check if it's a simple line or a multiline.
                            print 'Adding node ', idx + 1
                            l_inter_coords=np.array(l_inter.coords)
                            idx+=1 
                            idxs.append(idx)
                            G.G_nav.add_node(idx, coord=(l_inter_coords[0] + l_inter_coords[1])/2., sec=s) # add a node in s on the piece of line.
                            assert G.polygons[s].contains(Point((l_inter_coords[0] + l_inter_coords[1])/2.))
                            change=True 
                        else:
                            raise Exception('Problem: multi line detected.')
                    for idxx in idxs: # Detach the two nodes, attach all intermediate created nodes together.
                        n1,n2=tuple(sorted([idx2 for idx2 in idxs + [nav,neigh] if idx2!=idxx],\
                            key= lambda idx2:np.linalg.norm(np.array(G.G_nav.node[idxx]['coord']) - np.array(G.G_nav.node[idx2]['coord'])))[:2])
                        
                        G.G_nav.add_edge(idxx,n1)
                        G.G_nav.add_edge(idxx,n2)
                    G.G_nav.remove_edge(nav,neigh)           
               
    if generate_new_navpoints: #Check
        #G.G_nav.clean_borders()
        for nav in G.G_nav.nodes():
            sec1=G.G_nav.node[nav]['sec']
            for neigh in G.G_nav.neighbors(nav):
                sec2=G.G_nav.node[neigh]['sec']
                if sec1<sec2 and not G[sec1].has_key(sec2):   
                    sects=[s for s in G.nodes() if s!=sec1 and s!=sec2 and \
                    #G.neighbors(sec1) if s in G.neighbors(sec2) \
                         LineString([G.G_nav.node[nav]['coord'], G.G_nav.node[neigh]['coord']]).intersects(G.polygons[s])]
                    raise Exception('Problem:',sec1, sec2, sects, nav, neigh, G.G_nav.node[nav]['coord'], G.G_nav.node[neigh]['coord'])

    # We check if every couples of neighbors (sectors) have at least one couple of navpoints which are neighbors.
    G, problem = check_everybody_is_attached(G, repair=True)
    G, problem = check_everybody_is_attached(G, repair=False)
    assert not problem

    # We check if, within each sectors, there is only one connected component
    G, problem = check_everybody_has_one_cc(G, repair=True)
    G, problem = check_everybody_has_one_cc(G, repair=False)
    assert not problem
                
    #G.G_nav.show(stack=True, colors=[_colors[G.G_nav.node[n]['sec']%len(_colors)] for n in G.G_nav.nodes()])
    #show_everything(G.polygons,G)    
    #plt.show()
            
    ########## Generate Capacities and weights ###########

    if mode == 'real':
        flows, times, flights_selected = extract_flows_from_data(paras_real, [G.G_nav.node[n]['name'] for n in G.G_nav.nodes()])
        G.flights_selected = flights_selected
        print 'Selected at first:', len(flights_selected), 'flights.'
    else:
        flights_selected = []

    G = give_capacities_and_weights(G, mode, paras_G)
    
    ############# Computing shortest paths ###########
    
    G.Nfp=paras_G['Nfp']   
    G.G_nav.Nfp=G.Nfp
    
    print 'Computing shortest_paths (sectors) ...'
    #G.initialize_load()
    pairs_deleted = G.compute_shortest_paths(G.Nfp, repetitions=False, old=False)   
    G.infer_airports_from_short_list()
    #G.G_nav.infer_airports_from_sectors(G.airports, paras_G['min_dis'])

    pairs = G.G_nav.short.keys()[:]
    for (p1,p2) in pairs:
        s1=G.G_nav.node[p1]['sec']
        s2=G.G_nav.node[p2]['sec']
        if (s1, s2) in pairs_deleted:
            del G.G_nav.short[(p1,p2)]
            #G.G_nav.pairs.remove((p1,p2))
            print 'I removed the pair of navpoints', (p1,p2), ' because the corresponding pair', (s1, s2), ' has been removed.'

    G.G_nav.infer_airports_from_short_list()

    #G=compute_sp_restricted(G, G.Nfp, silent=False)
     
    print 'Computing shortest_paths (navpoints) ...'
    print 'Number of pairs before computation:', len(G.G_nav.short.keys())  
    G.compute_sp_restricted(G.Nfp, silent = True)
    G.G_nav.infer_airports_from_short_list()
    G.check_repair_sector_airports()

    print 'Number of pairs before computation:', len(G.G_nav.short.keys())
    # with silence(False):
    #     G.G_nav.compute_pairs_based_on_short(paras_G['min_dis'])
    G.check_airports_and_pairs()

    if mode == 'real': 
        # we remove from the selected flights all those which are not valid anymore, because of the airports we deleted.
        fl_s = G.flights_selected[:]

        for f in fl_s:
            if not (G.G_nav.idx_navs[f['route_m1'][0][0]], G.G_nav.idx_navs[f['route_m1'][-1][0]]) in G.G_nav.short.keys():
                G.flights_selected.remove(f)
                print 'I remove a flights because it was flying from', G.G_nav.idx_navs[f['route_m1'][0][0]], 'to', G.G_nav.idx_navs[f['route_m1'][-1][0]]

        # readjust capacities and weights based on the new set of flights
        G = give_capacities_and_weights(G, mode, paras_G)
        print 'Selected finally', len(G.flights_selected)

        G.check_all_real_flights_are_legitimate()

    print 'Selected finally', len(G.G_nav.airports), 'airports (navpoints) and', len(G.G_nav.short.keys()), 'pairs.'

    #G.G_nav.compute_shortest_paths(G.G_nav.Nfp)
    
    ##################### Name #######################
    
    long_name=G.type_of_net + '_N' + str(len(G.nodes()))
    
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
    
    with open(name + '.pic','w') as f:
        pickle.dump(G, f)

    print 'Network saved as', name + '.pic'
    #show_everything(G.polygons,G,save=True,name=name,show=False)       
    
    print 'Number of sectors:', len(G.nodes())
    print 'Number of navpoints:', len(G.G_nav.nodes())
    if mode == 'real':
        print 'Number of flights selected:', flights_selected
    print 'Done.'
        
    #draw_network_map(G.G_nav, title=name, trajectories=G.G_nav.short.values()[0])
    # for p in G.G_nav.short.values()[0]:
    #     print p
    draw_network_and_patches(None, G.G_nav, G.polygons, name=name, show=True, flip_axes=True, trajectories=G.G_nav.short.values()[0])
    return G
    
if  __name__=='__main__':
    country_='LF'
    airac_='334'
    #type_of_net='D'                 #type of graph ('D'=Delaunay triangulation, 'T'=triangular lattice, "E"=Erdos-Renyi random graph)
    #N=30                            #order of the graph (in the case net='T' verify that the order is respected)        
    airports_=[65,20]                #IDs of the nodes used as airports
    #airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
    #airports=[65,22,10,45, 30, 16]
    #airports=[]
    nairports_=len(airports_)         #0 for all airports (real network)                  #number of airports
    nairports_=0#0                  #number of airports
    pairs_=[]#[(22,65)]              #available connections between airports
    #[65,20]
    #[65,22]
    #[65,62]
    min_dis_=2                       #minimum number of nodes between two airpors
    airports_from_data_=True#True
    
    Nfp_=10
    
    generate_weights_=True
    typ_weights_='coords'
    sigma_=0.01
    generate_capacities_=False
    typ_capacities_='constant'
    C_=5                             #sector capacity
    C_airport_=20
    compute_areas_=True
    N_by_sectors_=10
    #suppl_par_capacity=[0.3]
    suppl_par_capacity_=['sqrt']

    mode_ = 'real' #real for real network, fake for generated network.

    layer_ = 350
    
    file_capacities_ = 'capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'

    paras_real_ = get_paras()
    paras_real_['zone'] = country_
    paras_real_['airac'] = airac_
    paras_real_['type_zone'] ='EXT'
    paras_real_['filtre'] ='Weak'
    paras_real_['mode'] ='navpoints'
    paras_real_['cut_alt'] = 240.
    paras_real_['both'] = False
    paras_real_['n_days'] = 1
    
    #name='DEL29_C5_44_26_Nfp10_cross5_natural_sector_paths_ICA'#_borders_clean' ICA=infinit capacity airport
    #name='DEL29_C5_65_20_v2'#_borders_clean' ICA=infinit capacity airport
    name_ = country_ + '29_RC' + (layer_!=0)*('_FL' + str(layer_)) + airports_from_data_*('_DA' + str(nairports_)) + (nairports_!=0)*('_seed' +str(see_)) + 'v3_' + paras_real_['filtre'] #DA stands for Data airports (airports from data).
    
    paras_G={k[:-1]:v for k,v in vars().items() if k[-1]=='_' and k[0]!='_' and k!='version'}
    
    print 'paras_G:', paras_G

    G = prepare_navpoint_network(paras_G, mode = mode_, generate_new_navpoints=False, generate_new_sectors=False, layer=layer_, airports_from_data=airports_from_data_)