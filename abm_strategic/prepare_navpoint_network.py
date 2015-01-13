#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

===========================================================================
This file is used to build a navpoint network with superimposed sector 
network. The main function is prepare_hybrid_network, with which the 
user can build a totally new network, or take some data from elsewhere.
===========================================================================
"""

import sys
sys.path.insert(1, '..')
import pickle
import networkx as nx
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union, unary_union
from random import choice, seed, sample
from string import split
from copy import deepcopy
from os.path import join

from simAirSpaceO import Net, NavpointNet
from utilities import clean_network

#Distance
from libs.tools_airports import get_paras, extract_flows_from_data, expand#, get_flights
#Modules
from libs.general_tools import draw_network_and_patches, silence, counter, delay, date_st, make_union_interval

if 0:
    # Manual seed
    see_=1
    #see_ = 15 #==> the one I used for new, old_good, etc.. 
    # see=2
    print "===================================="
    print "USING SEED", see_
    print "===================================="
    seed(see_)

version='2.9.8'

_colors = ('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')


def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def attach_termination_nodes(G):
    """
    New in 2.9.6: attach nodes with degree 1 to closest node in the same sector.
    Changed in 2.9.7: do not attach airports
    """
    changed = True
    while changed:
        changed = False
        for n in G.G_nav.nodes():
            sec = G.G_nav.node[n]['sec']
            if G.G_nav.degree(n)==1 and not n in G.G_nav.get_airports():
                changed = True
                pairs = [(n, n2) for n2 in G.node[sec]['navs'] if n2!=n and (not n2 in G.G_nav.neighbors(n)) and (not n in G.G_nav.navpoints_borders or not n2 in G.G_nav.navpoints_borders)]
                distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                n1_selected, n2_selected = pairs[np.argmin(distances)]
                print "I attached node", n1_selected, "with degree 1 to node", n2_selected
                G.G_nav.add_edge(n1_selected, n2_selected)
    return G

def attach_two_sectors(s1, s2, G):
    """
    Attach two sectors by attaching two navpoints in each of them.
    The closest navpoints to each other are chosen.
    """
    navs_in_s1=G.node[s1]['navs']#[n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s1]
    try:
        assert len(navs_in_s1)>0
    except AssertionError:
        print "There is no navpoint in sector", s1, "!"
        raise
    navs_in_s2=G.node[s2]['navs']#[n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s2]
    try:
        assert len(navs_in_s2)>0
    except AssertionError:
        print "There is no navpoint in sector", s2, "!"
        raise
    
    pairs=[(n1, n2) for n1 in navs_in_s1 for n2 in navs_in_s2]
    #print "Pairs:", pairs
    distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
    #print "Distances:", distances
    n1_selected, n2_selected = pairs[np.argmin(distances)]
    G.G_nav.add_edge(n1_selected,n2_selected, weight=np.linalg.norm(np.array(G.G_nav.node[n1_selected]['coord']) - np.array(G.G_nav.node[n2_selected]['coord'])))
    return G

def automatic_name(G, paras_G):
    long_name=G.type_of_net + '_N' + str(len(G.nodes()))
    
    if G.airports!=[] and len(G.airports)==2:
       long_name+='_airports' +  str(G.airports[0]) + '_' + str(G.airports[1])
    elif len(G.airports)>2:
        long_name+='_nairports' + str(len(G.airports))
    if paras_G['pairs']!=[] and len(G.airports)==2:
        long_name+='_direction_' + str(paras_G['pairs'][0][0]) + '_' + str(paras_G['pairs'][0][1])
    long_name+='_cap_' + G.typ_capacities
    
    if G.typ_capacities!='manual':
        long_name+='_C' + str(paras_G['C'])
    long_name+='_w_' + G.typ_weights
    
    if G.typ_weights=='gauss':
        long_name+='sig' + str(paras_G['sigma'])
    long_name+='_Nfp' + str(G.Nfp)

    return long_name

def check_and_fix_empty_sectors(G, checked, repair=False):
    # Check if every sectors have at least one navpoint left. 
    # TODO: another way to go would be to add a constant number of navpoints per sector. On the other
    # hand, having exactly the same number of navpoints per sector might not be really realistic.
    problem = False
    try:
        empty_sectors = [k for k,v in checked.items() if not v]
        assert len(empty_sectors)==0
    except AssertionError:
        print "The following sectors do not have any navpoint left:", empty_sectors
        if repair:
            print "I add a navpoint at the centroid for them."
            print "I add also some links with the closest points of neighboring sectors."
            for sec in empty_sectors:
                nav = len(G.G_nav.nodes())
                print G.polygons
                RP = G.polygons[sec].representative_point()
                print RP.wkt
                coords = RP.coords
                print coords
                G.G_nav.add_node(nav, coord = coords)

                if G.polygons[sec].contains(Point(np.array(coords))):
                    G.G_nav.node[nav]['sec']=sec
                    G.node[sec]['navs'].append(nav)
                else:
                    Exception("The representative point in not in the shape!")
                
                for sec2 in G.neighbors(sec):
                    pairs = []
                    for nav2 in G.node[sec2]['navs']:
                        pairs.append((nav, nav2))
                    distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                    n1_selected, n2_selected = pairs[np.argmin(distances)]
                    G.G_nav.add_edge(n1, n2, weight=np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])))
        problem = True
    except:
        raise

    return G, problem

def check_empty_polygon(G, repair = False):
    for n in G.nodes():
        try:
            assert len(G.polygons[n].representative_point().coords) != 0 
        except AssertionError:
            print "Sector", n, "has no surface. I delete it from the network."
            if repair:
                G.remove_node(n)
                del G.polygons[n]

def check_everybody_is_attached(G, repair=False):
    """
    Check if all neighboring sectors have at least one navpoint each which are neighbors.
    """
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

def check_everybody_has_one_cc(G, repair=False):
    """
    Changed in 2.9.6: we attach the nodes to the closest. Recursive check
    """
    problem=False
    for s in G.nodes():
        nodes_in_s = [n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s]
        H_nav_s = G.G_nav.subgraph(nodes_in_s)
        cc = nx.connected_components(H_nav_s)
        while len(cc)>1:
            print 'Problem: sector', s, 'has more than one connected component (' + str(len(cc)) + ' exactly).'
            #cc.sort(key=lambda c:len(c))
            #cc = cc[::-1] # sort the components by decreasing size
            problem=True
            if repair:
                print "I'm fixing this."
                c1 = cc[0] # we attach everyone to the biggest cc.
                #for j in range(len(cc)):
                #c1 = cc[j]
                all_other_nodes = [n for c in cc for n in c if c!=c1]
                pairs=[(n1, n2) for n1 in c1 for n2 in all_other_nodes]
                distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                n1_selected, n2_selected = pairs[np.argmin(distances)]
                w=np.linalg.norm(np.array(G.G_nav.node[n1_selected]['coord']) - np.array(G.G_nav.node[n2_selected]['coord']))
                G.G_nav.add_edge(n1_selected,n2_selected, weight=w)
                H_nav_s.add_edge(n1_selected,n2_selected, weight=w)
            cc = nx.connected_components(H_nav_s)

    return G, problem

def check_matching(G, repair=False):
    print "Checking matching integrity..."
    problem = False
    for s in G.nodes():
        for n in G.node[s]['navs']:
            if not G.G_nav.node[n].has_key('sec'):
                problem = True
                print "Navpoint", n, "do not have sector", s, "matched whereas sector", s,\
                 "has navpoint", n, "in the list of navs."
                if repair:
                    print "I match", n, "to", s, "."
                    G.G_nav.node[n]['sec']=s
            elif G.G_nav.node[n].has_key('sec') and G.G_nav.node[n]['sec']!=s:
                raise Exception("Navpoint", n, "have matches sector", G.G_nav.node[n]['sec'], \
                 "instead of sector", s, ".")

    for n in G.G_nav.nodes():
        if not G.G_nav.node[n].has_key('sec'):
            raise Exception("Navpoint", n, "do not have a matching sector.")
        else:
            try:
                s = G.G_nav.node[n]['sec']
                assert n in G.node[s]['navs']
            except AssertionError:
                problem = True
                print "Sector", s, "do not match navpoint", n, "whereas navpoint", n,\
                 "matches sector", s, "."
                if repair:
                    print "I match", s, "to", n
                    G.node[s]['navs'].append(n)

    return G, problem

def check_two_sectors_attached(s1, s2, G):
    """
    Check if two sectors s1 and s2 have at least one navpoint each which are neighbors.
    """
    found=False
    for n1, n2 in G.G_nav.edges():
        if (G.G_nav.node[n1]['sec']==s1 and G.G_nav.node[n2]['sec']==s2) or (G.G_nav.node[n1]['sec']==s2 and G.G_nav.node[n2]['sec']==s1):
            found=True
            break
    return found

def compute_navpoints_borders(borders_coordinates, shape, lin_dens = 10, only_outer_boundary = False, thr = 1e-2):
    """
    Put some navpoints on each segments given by borders.
    New in 2.9.7.
    """

    navpoints = []
    small = 10**(-5.)
    for c1,c2 in borders_coordinates:
        c1, c2 = np.array(c1), np.array(c2)
        #print "c1=", c1, "; c2=", c2
        a = c2 - c1
        l = np.linalg.norm(a) #length of border
        n_p = max(1,int(l*lin_dens)) # number of points to put on the border
        d = l/float(n_p+1) # distance between each future pair of nodes.
        #print "a=", a, "; l=", l, "; n_p=", n_p, "; d=", d
        k = 1
        changed=False
        while d*k<l-10**(-4.):# put points until vertex is reached
            #We shift slighly the points on the borders so there is no ambiguity about which sector they belong to.
            for shift in [np.array([s*small, ss*small]) for s in [1., -1.] for ss in [1., -1.]]: 
                c = c1 + k*(d/l)*a + shift
                #print "c1, k*(d/l)*a, shift", c1, k*(d/l)*a, shift
                P = Point(c)
                if shape.contains(P): # Check that this point is actually in the global shape.
                    break
                #print "With these coordinates:", c, "the point is not contained in the global shape."
            #if not shape.contains(P):
            #    print "Could not find any coordinates in the shape for this point."

            if -1.<c[0]<1. and -1.<c[1]<1.:
                if not only_outer_boundary or shape.exterior.distance(P)<thr:
                    changed = True
                    navpoints.append(c)
                    #print "New border point with this coordinate:", c
                    try:
                        assert shape.contains(P)
                    except AssertionError:
                        print "The selected node is not contained in the global shape."
                        raise
            k+=1

    return navpoints

def compute_possible_outer_pairs(G):
    pairs = []

    for i, n1 in enumerate(G.G_nav.outer_nodes):
        s1 = G.G_nav.node[n1]['sec']
        for j in range(i+1,len(G.G_nav.outer_nodes)):
            n2 = G.G_nav.outer_nodes[j]
            s2 = G.G_nav.node[n2]['sec']
            if s1!=s2 and not s2 in G.neighbors(s1):
                pairs.append([n1,n2])
                pairs.append([n2,n1])

    return pairs

def compute_voronoi(G):
    polygons={}
    nodes = G.nodes()
    vor = Voronoi(np.array([G.node[n]['coord'] for n in nodes]))
    for i,p in enumerate(vor.point_region):
        r = vor.regions[p]
        coords=[vor.vertices[n] for n in  r + [r[0]] if n!=-1]
        if len(coords)>2:
            G.node[i]['area']=area(coords)
            polygons[i]=Polygon(coords)
            try:
                assert abs(G.node[i]['area'] - polygons[i].area)<10**(-6.)
            except:
                raise Exception(i, G.node[i]['area'], polygons[i].area)
        else:
            print "Problem: the sector", i, "has the following coords for its vertices:", coords
            print "I remove the corresponding node from the network."
            G.remove_node(nodes[i])
    
    eps=0.1
    minx,maxx,miny,maxy=min([n[0] for n in vor.vertices]) -eps, max([n[0] for n in vor.vertices]) +eps, min([n[1] for n in vor.vertices]) -eps, max([n[1] for n in vor.vertices]) +eps
    
    square=Polygon([[-1,-1],[-1,1],[1,1],[1,-1],[-1,-1]])

    for n,pol in polygons.items():
        polygons[n]=pol.intersection(square)
        
    G.polygons=polygons
    return G, vor

def default_prepare_sectors_network(paras_G, generate_new_sectors=False, mode = "fake", no_airports=False):
    """
    New in 2.9.6: refactorization.
    Obsolete.
    """
    file_net = None
    if not generate_new_sectors:
        if mode == 'fake':
            file_net = 'DEL_C_6A.pic'
            #G.type_of_net='D'
        else:
            #fille='Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0_undirected_threshold'
            file_net='../Modules/' + paras_G['country'] + '_sectors_network_one_layer_FL' + str(layer) + '_A334.pic'

    file_polygons = None

    if mode!="fake":
        file_polygons = '../Modules/sectors_network_one_layer_FL' + str(layer) + '_A334_polygons.pic'

    paras_G['file_net'] = file_net
    paras_G['file_polygons'] = file_net

    return prepare_navpoint_network(paras_G, no_airports=no_airports)

def detect_nodes_on_boundaries(paras_G, G, thr = 1e-2):
    """
    Detects the nodes on the OUTER boundary of the airspace.
    New in 2.9.6.
    """
    if paras_G['make_borders_points']:
        shape = G.global_shape
        nodes = G.G_nav.navpoints_borders

        outer_nodes = []
        for n in nodes:
            coords = G.G_nav.node[n]['coord']
            P = Point(np.array(coords))
            if shape.exterior.distance(P)<thr:
                outer_nodes.append(n)

        G.G_nav.outer_nodes = outer_nodes
    else:
        G.G_nav.outer_nodes = G.G_nav.navpoints_borders

    return G

def erase_small_sectors(G, thr):
    """
    New in 2.9.6: remove sectors having a small number of navpoints.
    Changed in 2.9.7: set thr<0 for no action (instead of 0).
    """
    #if thr>0:
    removed = []
    for s in G.nodes()[:]:
        if len(G.node[s]['navs'])<=thr:
            for np in G.node[s]['navs']:
                G.G_nav.remove_node(np)
            G.remove_node(s)
            removed.append(s)

    return G, removed

def extract_airports_from_traffic(G, flows): #,paras_G):
    """
    New in 2.8: gives naiports "airports" among all points of entries and exits from data trajectories.
    Changed in 2.9: gives pairs as well. Gives all airports.
    Changed in 2.9.4: added paras_real.
    Chanded in 2.9.5: flows are given externally
    Obsolete probably TODO.
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

    #G.add_airports([G.idx_nodes[airports, paras_G['min_dis'])

    return [G.G_nav.idx_nodes[name] for name in all_airports], [(G.G_nav.idx_nodes[p[0]], G.G_nav.idx_nodes[p[1]]) for p in all_pairs]
    
def extract_weights_from_traffic(G, flights):
    """
    For each edge of G, computes the average time elapsed between two navpoints.
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: flights are given externally.
    """
    print 'Extracting weights from data...'

    #flights=get_flights(paras_real)
    weights = {}
    pop = {}
    for f in flights:
        r = f['route_m1t']
        for i in range(len(r)-1):
            if G.idx_nodes.has_key(r[i][0]) and G.idx_nodes.has_key(r[i+1][0]):
                p1 = G.idx_nodes[r[i][0]]
                p2 = G.idx_nodes[r[i+1][0]]
                if G.has_edge(p1,p2):
                    weights[(p1,p2)] = weights.get((p1, p2),0.) + delay(np.array(r[i+1][1]),starting_date=np.array(r[i][1]))/60.
                    pop[(p1,p2)] = pop.get((p1, p2),0.) + 1

    for k in weights.keys():
        weights[k] = weights[k]/float(pop[k])

    if len(weights.keys())<len(G.edges()):
        print 'Warning! Some edges do not have a weight!'
    return weights

def extract_capacity_from_traffic(G, flights, date = [2010, 5, 6, 0, 0, 0]):
    """
    New in 2.9.3: Extract the "capacity", the maximum number of flight per hour, based on the traffic.
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: fligths are given externally.
    """
    print 'Extracting the capacities from traffic...'

    weights={}
    pop={}
    loads = {n:[0 for i in range(48)] for n in G.nodes()}  # why 48? TODO.
    for f in flights:
        hours = {}
        r = f['route_m1t']
        for i in range(len(r)):
            if G.G_nav.idx_nodes.has_key(r[i][0]):
                p1=G.G_nav.idx_nodes[r[i][0]]
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
            # print 'List of idx of navpoints:', [G.G_nav.idx_nodes[n[0]] for n in f['route_m1t'] if n[0] in G.G_nav.idx_nodes.keys()]
            # print 'List of idx of sectors:', [G.G_nav.node[G.G_nav.idx_nodes[n[0]]]['sec'] for n in f['route_m1t'] if n[0] in G.G_nav.idx_nodes.keys() and G.G_nav.has_node(G.G_nav.idx_nodes[n[0]])]
            
            navs = [n[0] for n in f['route_m1t']]
            navs_idx = [G.G_nav.idx_nodes[n] for n in navs if G.G_nav.idx_nodes.has_key(n)]
            sects_idx = [G.G_nav.node[n]['sec'] for n in navs_idx if G.G_nav.has_node(n)]
            sects = [G.node[s]['name'] for s in sects_idx if G.has_node(s)]
            #sects=[G.G_nav.node[G.G_nav.idx_nodes[n[0]]]['sec']] for n in f['route_m1t'] if n[0] in G.G_nav.idx_nodes.keys()\
            # and G.G_nav.has_node(G.G_nav.idx_nodes[n[0]])]
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
        capacities[G.idx_nodes[n]]=max([p[1] for p in make_union_interval(intervals[n])])
    
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

def give_capacities_and_weights(G, paras_G):
    if paras_G['generate_capacities_from_traffic']:
        capacities = extract_capacity_from_traffic(G, paras_G['flights_selected'])
        G.fix_capacities(capacities)
    else:
        if paras_G['capacities']==None:
            G.generate_capacities(typ=paras_G['typ_capacities'], C=paras_G['C'], par=paras_G['suppl_par_capacity'])
            #G.typ_capacities=paras_G['typ_capacities']
        else:
            G.fix_capacities(paras_G['capacities'])

    for n in G.nodes():
        try:
            assert G.node[n].has_key('capacity')
        except AssertionError:
            print "This node did not receive any capacity:", n
            raise
        except:
            raise


    if paras_G['generate_weights_from_traffic']:
        weights = extract_weights_from_traffic(G.G_nav, paras_G['flights_selected'])
        G.G_nav.fix_weights(weights, typ='traffic')
        avg_weight = np.mean([G.G_nav[e[0]][e[1]]['weight'] for e in G.G_nav.edges() if G.G_nav[e[0]][e[1]].has_key('weight')])
        for e in G.G_nav.edges():
            if not G.G_nav[e[0]][e[1]].has_key('weight'):
                print "This edge did not receive any weight:", e, ", I set it to the average(", avg_weight, ")"
                G.G_nav[e[0]][e[1]]['weight'] = avg_weight
    else:
        if paras_G['weights']==None:
            G.G_nav.generate_weights(typ='coords', par=paras_G['par_weights'])
        else:
            G.G_nav.fix_weights(paras_G['weights'], typ='data')
    
    for e in G.G_nav.edges():
        try:
            assert G.G_nav[e[0]][e[1]].has_key('weight')
        except AssertionError:
            print "This edge did not receive any weight:", e
            raise
        except:
            raise    

    G.generate_weights(typ='coords', par=20.) #That's for sector network... should go in the preparation of the sector network probably.
    G.build_H() #This is the graph used for shortest_path
    G.G_nav.build_H()

    return G

def navpoints_at_borders(G, lin_dens=10., only_outer_boundary = False, thr = 1e-2):
    """
    Changed in 2.9.7: don't use vor anymore. More efficient.
    """
    shape = G.global_shape
    if only_outer_boundary:
        borders_coords = list(shape.exterior.coords)
        borders = [(borders_coords[i], borders_coords[i+1]) for i in range(len(borders_coords)-1)]
    else:
        borders = []
        for n, pol in G.polygons.items():
            for i in range(len(pol.exterior.coords)-1):
                seg = (pol.exterior.coords[i], pol.exterior.coords[i+1])
                if not (seg in borders or seg[::-1] in borders): # this is to ensure non-redundancy of the borders.
                    borders.append(seg)

    navpoints = compute_navpoints_borders(borders, shape, lin_dens = lin_dens, only_outer_boundary = only_outer_boundary, thr = thr)

    print "I will create", len(navpoints), "navpoints on the borders."
    return navpoints

def _OLD_navpoints_at_borders(G, lin_dens=10., only_outer_boundary = False, thr = 1e-2):
    """
    Create navpoints at the bondaries of a Voronoi tesselation.
    Changed in 2.9.7: added the possibility to add points only on the outer bounday. Changed 
    the small shift if the point lies outside the global shape.
    """
    small = 10**(-5.)
    borders=[]
    shape = G.global_shape

    print "shape exterior coordinates:", list(shape.exterior.coords)
    for p in G.vor.point_region:
        r=G.vor.regions[p]
        for n1,n2 in [(r[i],r[i+1]) for i in range(len(r)-1)] + [(r[-1],r[0])]:
            borders.append(tuple(sorted((n1,n2))))
    
    navpoints=[]
    for n1,n2 in borders: # TODO: this loop is HIGHLY inefficient...
        a=np.array(G.vor.vertices[n2] - G.vor.vertices[n1])
        l=np.linalg.norm(a) #length of border
        n_p=max(1,int(l*lin_dens)) # number of points to put on the border
        d=l/float(n_p+1) # distance between each future pair of nodes.
        k=1
        changed=False
        while d*k<l-10**(-4.):# put points until vertex is reached
            #We shift slighly the points on the borders so there is no ambiguity about which sector they belong to.
            for shift in [[s*small, ss*small] for s in [1., -1.] for ss in [1., -1.]]: 
                x,y=list(G.vor.vertices[n1] + k*(d/l)*a +[small, small])
                if shape.contains(Point(np.array([x,y]))): # Check that this point is actually in the global shape.
                    break
                print "With this shift:", shift, "the point is not contained in the global shape."

            if not shape.contains(Point(np.array([x,y]))):
                print "WARNING: The selected node is not contained in the global shape." # Or exception?

            if -1.<x<1. and -1.<y<1.:
                print "Distance of point to closest outer boundary:", shape.exterior.distance(Point(np.array([x,y])))
                if not only_outer_boundary or shape.exterior.distance(Point(np.array([x,y])))<thr:
                    changed=True
                    navpoints.append([x,y])
                    print "New border point with this coordinate:", [x,y]
            k+=1
        #if not changed:
        #    print 'Did not put any points between:', n1,n2, G.vor.vertices[n1], G.vor.vertices[n2]

    print "Final number of point on the outer boundaries:", len(navpoints)
    return navpoints

def prepare_sectors_network(paras_G, no_airports=False):  
    """
    New in 2.9.6: refactorization.
    """
    G=Net()
    G.type='sec' #for sectors
    G.type_of_net = paras_G['type_of_net']

    if paras_G['net_sec'] !=None:
        G.import_from(paras_G['net_sec'], numberize=((type(paras_G['net_sec'].nodes()[0])!=type(1.)) and (type(paras_G['net_sec'].nodes()[0])!=type(1))))
    else:
        G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'],put_nodes_at_corners = True)

    if paras_G['polygons']!=None:
        G.polygons={}
        for name, shape in paras_G['polygons'].items():
            G.polygons[G.idx_nodes[name]]=shape
    else:
        G, vor= compute_voronoi(G)
        #G.vor = vor #I should remove this. It is useful for navpoints_at_borders only, and I am sure I could do without it.

    
    # Check if every sector has a polygon
    for n in G.nodes():
        try:
            assert G.polygons.has_key(n)
        except:
            print "Sector", n, "doesn't have any polygon associated!"
            raise

    check_empty_polygon(G, repair = True)
    check_empty_polygon(G, repair = False)

    recompute_neighbors(G)

    G.global_shape=unary_union(G.polygons.values()).convex_hull

    if not no_airports:
        if paras_G['airports']!=[]:
            G.add_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
        else:
            G.generate_airports(paras_G['nairports'],paras_G['min_dis'])

    return G
        
def prepare_sectors_network_bis(paras_G, mode = 'fake', generate_new_sectors=False, layer=0, no_airports=False):  
    """
    Prepare a sector network.
    Changed in 2.9.6: obsolete.
    """   
    G=Net()
    G.type='sec' #for sectors

    if generate_new_sectors:
        G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'])
    else:
        if mode == 'fake':
            fille='DEL_C_6A'
            G.type_of_net='D'
        else:
            #fille='Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0_undirected_threshold'
            fille='../Modules/' + paras_G['country'] + '_sectors_network_one_layer_FL' + str(layer) + '_A334'
            G.type_of_net='R'  # For "Real"
            G.airac=paras_G['airac']
            G.country=paras_G['country']
            G.layer=layer #indicates the layer; '0' for full otherwise.
   
        with open(fille + '.pic','r') as _g:           
            H=pickle.load(_g)
        
        if G.type_of_net=='R':  # Numberize the sectors (instead of initial names)
            G.idx_nodes={s:i for i,s in enumerate(H.nodes())}
            for n in H.nodes():
                G.add_node(G.idx_nodes[n], name=n, **H.node[n])
            for e in H.edges():
                G.add_edge(G.idx_nodes[e[0]], G.idx_nodes[e[1]])
        else:
            G.import_from(H)

        if not no_airports:
            if paras_G['airports']!=[]:
                G.add_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
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
                G.polygons[G.idx_nodes[name]]=shape
            G.global_shape=cascaded_union(G.polygons.values())
                
    return G
    
def recompute_neighbors(G):
    """
    New in 2.9.6: Checks if neighbouring sectors have a common boundary. Disconnects them otherwise.
    """

    for n1 in G.nodes():
        neighbors = G.neighbors(n1)[:]
        for n2 in neighbors:
            if n1<n2:
                try:
                    assert G.polygons[n1].touches(G.polygons[n2])
                except AssertionError:
                    print "The two sectors", n1, "and", n2, "are neighbors but does not touch each other. I cut the link."
                    G.remove_edge(n1,n2)

def reduce_airports_to_existing_nodes(G, pairs, airports):
    if pairs!=None:
        for e1, e2 in pairs[:]:
            if not G.has_node(e1) or not G.has_node(e2):
                pairs.remove((e1,e2))
    if airports!=None:
        for a in airports[:]:
            if not G.has_node(a):
                airports.remove(a)        
    return pairs, airports

def remove_singletons(G, pairs_nav):
    for n1, n2 in pairs_nav[:]:
        if G.G_nav.node[n1]['sec'] == G.G_nav.node[n2]['sec']:
            pairs_nav.remove((n1, n2))

    return pairs_nav

def segments(p):
    return zip(p, p[1:] + [p[0]])
    
def show_polygons(polygons,nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for pol in polygons.values():
        patch = PolygonPatch(pol,alpha=0.5, zorder=2)
        ax.add_patch(patch) 
    plt.plot([n[0] for n in nodes], [n[1] for n in nodes],'ro')
    #plt.show()
    
def show_everything(polygons,G,save=True,name='network_navpoints', show=True):
    draw_network_and_patches(G,G.G_nav,polygons, save=save, name=name,show=show)

def write_down_network(G):
    """
    New in 2.9.4: write down the atributes, nodes and edges of the sector network.
    Used in particular to convert 2.9 graphs (model 2) to 2.6 (model 1).
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
        p_nav = [G.G_nav.idx_nodes[p[0]] for p in f['route_m1']]
        p_sec = G.convert_path(p_nav)

        entry_times=[0.]*(len(p_sec)+1)
        entry_times[0] = f['route_m1t'][0][1]

        road = delay(f['route_m1t'][0][1])
        sec = G.G_nav.node[p_nav[0]]['sec']
        j=0
        for i in range(1,len(p_nav)):
            w = delay(f['route_m1t'][i][1]) - delay(f['route_m1t'][i-1][1])  #G.G_nav[p_nav[i-1]][p_nav[i]]['weight']
            if G.G_nav.node[p_nav[i]]['sec']!=sec:
                j+=1
                entry_times[j] = date_st(road + w/2.)
                sec=G.G_nav.node[p_nav[i]]['sec']
            road += w
        entry_times[len(p_sec)] = date_st(road)

        f['sec_path'] = p_sec
        f['sec_path_t'] = entry_times

    with open(G.name + '_pieces2.pic', 'w') as f:
        dic = {}
        for k, a in G.graph.items():
            dic[k] = a

        for k in G.__dict__.keys():
            if not k in ['graph', 'short_nav', 'node', 'H', 'G_nav', 'edge', 'adj']:
                dic[k] = G.__dict__[k]

        pickle.dump(dic, f)

class NoEdges(Exception):
    pass

# =========================================================================================== #
# =========================================================================================== #

def prepare_hybrid_network(paras_G, rep='.', save=True, save_path=None, show=True):
    """
    New in 2.9.6: refactorization.
    Changed in 2.9.7: - navpoints are not contained in any sectors but touch one are linked to it.
                      - numberize also airports and pairs.
    Changed in 2.9.8: supports single sector network TODO. Supports custom external function 
        for choosing airports.
    TODO: make an object and different methods.
    """
    print

    ############ Prepare network of sectors #############
    print "Preparing network of sector..."
    G=prepare_sectors_network(paras_G, no_airports=paras_G['file_airports']==None)
    if len(G.edges())==0:
        raise NoEdges("The sector network has only one sector, this feature is not supported for now.")#TODO
    #G.show()
    print


    ############ Prepare nodes and edges of network of navpoints ###########
    print "Preparing nodes and edges of network of navpoints..."
    for n in G.nodes():
        G.node[n]['navs']=[]
        
    G.G_nav=NavpointNet()
    G.G_nav.type='nav'

    if paras_G['net_nav']==None:
        nav_at_borders = navpoints_at_borders(G, only_outer_boundary = not paras_G['make_borders_points'], lin_dens = paras_G['lin_dens_borders'])
        #nav_at_borders = []
        G.G_nav.build((paras_G['N_by_sectors']-1)*len(G.nodes()),paras_G['nairports'],paras_G['min_dis'], generation_of_airports=False, \
                sector_list=[G.node[n]['coord'] for n in G.nodes()], navpoints_borders=nav_at_borders, shortcut=0.01) #ATTENTION: min_dis is not the right one. TODO
        
    else:
        numberize = ((type(paras_G['net_nav'].nodes()[0])!=type(1.)) and (type(paras_G['net_nav'].nodes()[0])!=type(1)))
        G.G_nav.import_from(paras_G['net_nav'], numberize=numberize) 
        convert_minutes = max([abs(cc) for n in G.G_nav.nodes() for cc in G.G_nav.node[n]['coord']])>180. and max([abs(cc) for ccs in G.global_shape.exterior.coords for cc in ccs])<180
        if convert_minutes:
            print "I detected that the coordinates of navpoints were in minutes whereas coordinates of the area were in degree."
            print "I convert everything in degree."
            for n in G.G_nav.nodes():
                G.G_nav.node[n]['coord']=list(np.array(G.G_nav.node[n]['coord'])/60.) #Convert minutes in degrees
        for e in G.G_nav.edges(): # remove all edges which are geometrically outside of the global shape
            if not G.global_shape.contains(LineString([G.G_nav.node[e[0]]['coord'], G.G_nav.node[e[1]]['coord']])):
                print "I remove edge", e, "because it is outside the area."
                G.G_nav.remove_edge(*e)
        #G.G_nav, removed = restrict_to_connected_components(G.G_nav)
        #print "Removed the following nodes which were not connected to the biggest connected component:", removed
        G.G_nav, removed = clean_network(G.G_nav)
        print "Removed the following nodes with zero degree:", removed
        if numberize and paras_G['airports_nav']!=None and paras_G['pairs_nav']!=None:
            paras_G['airports_nav'] = [G.G_nav.idx_nodes[a] for a in paras_G['airports_nav']]
            paras_G['pairs_nav'] = [(G.G_nav.idx_nodes[e1], G.G_nav.idx_nodes[e2]) for e1, e2 in paras_G['pairs_nav']]

        G.G_nav.navpoints_borders = []

    print "Number of nodes in G_nav after preliminary preparation:", len(G.G_nav.nodes())
    print
    #G.G_nav.show(stack=True)

    for n in list(paras_G['airports_nav'])[:]:
        if not n in G.G_nav.nodes():
            paras_G['airports_nav'].remove(n)

    ######### Linking the two networks together ##########
    print "Linking the networks together..."
    not_found=[]
    nodes = G.nodes()
    checked = {sec:False for sec in nodes}
    for nav in G.G_nav.nodes(): # Finding to which sector each navpoint belongs.
        found=False
        i=0
        while not found and i<len(nodes):
            sec=nodes[i]
            P = Point(np.array(G.G_nav.node[nav]['coord']))
            if G.polygons.has_key(sec) and G.polygons[sec].contains(P):
                found=True
                checked[sec] = True
            i+=1
        i=0
        while not found and i<len(nodes):
            sec=nodes[i]
            P = Point(np.array(G.G_nav.node[nav]['coord']))
            if G.polygons.has_key(sec) and G.polygons[sec].touches(P):
                found=True
                checked[sec] = True
            i+=1
        if paras_G['expansion']>0.: # Expansion of the polygons
            n_steps = 10
            l = paras_G['expansion']/float(n_steps)
            while not found and l<paras_G['expansion']:
                i=0
                while not found and i<len(nodes):
                    sec=nodes[i]
                    try:
                        pol = expand(G.polygons[sec], l)                    
                        P = Point(np.array(G.G_nav.node[nav]['coord']))
                        if G.polygons.has_key(sec) and pol.contains(P):
                            found=True
                            checked[sec] = True
                    except AttributeError:
                        print "MultiPolygon detected for sector", sec, "!"
                    i+=1
                l+=paras_G['expansion']/float(n_steps)

        if not found:
            #print "I could not find a polygon for point:", nav, "of coordinates:", G.G_nav.node[nav]['coord']
            not_found.append(nav)
            # Remove node from network
            G.G_nav.remove_node(nav)
            # Remove node from list of airports
            if nav in paras_G['airports_nav']:
                #print "Removed", nav, "from list of airports."
                paras_G['airports_nav'].remove(nav)
            # Remove node from list of pairs
            for ee in paras_G['pairs_nav'][:]:
                if nav in set(ee):
                    #print "Removed", ee, "from list of pairs."
                    paras_G['pairs_nav'].remove(ee)
                
        else:
            G.G_nav.node[nav]['sec']=sec
            G.node[sec]['navs'].append(nav)
    print 'Navpoints for which I did not find a matching sector (I removed them from the network):', not_found

    # n = 0
    # print "Navpoint", n, ":", G.G_nav.node[n]['sec']
    # s = 1
    # print "Sector", s, ":", G.node[s]['navs']
    #G, problem = check_and_fix_empty_sectors(G, checked, repair = True)
    #G, problem = check_and_fix_empty_sectors(G, checked, repair = False)
    #G.G_nav.show(stack=True)

    G, problem = check_matching(G, repair=True)
    if problem: G, problem = check_matching(G, repair=False)
    G, removed = erase_small_sectors(G, paras_G['small_sec_thr']) # does nothing if paras_G['small_sec_thr']<0
    if len(removed)>0: print "Removed the following sectors because they had less than", paras_G['small_sec_thr']+1, "navpoint(s):", removed
    print 

    try:
        for n in paras_G['airports_nav']:
            assert n in G.G_nav.nodes()
    except:
        raise Exception("Airport", n, "is not in the list of nodes of G.G_nav")
    #draw_network_and_patches(None, G.G_nav, G.polygons, show=True, flip_axes=True, rep = rep)


    ########### Choose the airports #############
    print "Choosing the airports..."

    paras_G['pairs_nav'], paras_G['airports_nav'] = reduce_airports_to_existing_nodes(G.G_nav, paras_G['pairs_nav'], paras_G['airports_nav'])
    paras_G['pairs_sec'], paras_G['airports_sec'] = reduce_airports_to_existing_nodes(G, paras_G['pairs_sec'], paras_G['airports_sec'])

    if not paras_G['make_entry_exit_points']: 
        if paras_G['airports_sec']!=None and paras_G['airports_nav']!=None:
            # TODO
            print "Warning: specifiying airports for both sectors and navpoints is not operational."
            G.add_airports(paras_G['airports_sec'], -10, pairs=paras_G['pairs_sec'], C_airport = paras_G['C_airport'])
        elif paras_G['airports_sec']!=None and paras_G['airports_nav']==None:
            G.add_airports(paras_G['airports_sec'], -10, pairs=paras_G['pairs_sec'], C_airport = paras_G['C_airport'])
            G.G_nav.infer_airports_from_sectors(G.airports, paras_G['min_dis'])
        elif paras_G['airports_sec']==None and paras_G['airports_nav']!=None:
            if paras_G['function_airports_nav']!=None:
                # Custom function of airport building based on traffic.
                print "Extracting airports and pairs with custom function..."
                assert 'flights_selected' in paras_G.keys() and paras_G['flights_selected']!=None
                flights_selected, airports_nav, pairs_nav = paras_G['function_airports_nav'](G.G_nav, paras_G['flights_selected'])
                print "Deleted flights because no navpoints of the trajectories belonged to the network:", len(paras_G['flights_selected']) - len(flights_selected)
            else:
                airports_nav, pairs_nav = paras_G['airports_nav'], paras_G['pairs_nav']
            
            if not paras_G['singletons'] and pairs_nav!=None: 
                # Remove the pairs of navpoints which yield the same sec-entry and sec-exit.
                pairs_nav = remove_singletons(G, pairs_nav)
            G.G_nav.add_airports(airports_nav, -10, pairs=pairs_nav, C_airport = paras_G['C_airport'])
            G.infer_airports_from_navpoints(paras_G['C_airport'], singletons=paras_G['singletons'])
        else: # TODO: add generate nav-airports feature
            G.generate_airports(paras_G['nairports'], -10, C_airport=paras_G['C_airport'])
            G.G_nav.infer_airports_from_sectors(G.airports, paras_G['min_dis'])
    else:
        G = detect_nodes_on_boundaries(paras_G, G)
        pairs_outer = compute_possible_outer_pairs(G)
        print "Found outer nodes:", G.G_nav.outer_nodes
        print "With", len(pairs_outer), "corresponding pairs."
        print "I add them as airports."
        G.G_nav.add_airports(G.G_nav.outer_nodes, paras_G['min_dis'], pairs = pairs_outer)
        G.infer_airports_from_navpoints(paras_G['C_airport'])

    print 'Number of airports (sectors) at this point:', len(G.airports)
    print 'Number of connections (sectors) at this point:', len(G.connections())
    print 'Number of airports (navpoints) at this point:', len(G.G_nav.airports)
    print 'Number of connections (navpoints) at this point:', len(G.G_nav.connections()) 


    ############# Repair some stuff #############
    print "Repairing mutiple issues..."
    idx=max(G.G_nav.nodes())
    change=paras_G['file_net_nav']==None
    
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
               
    if paras_G['net_nav']==None: #Check if the previous operations went well.
        print "Cleaning borders..."
        G.G_nav.clean_borders()
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
    G, problem = check_everybody_is_attached(G, repair=True) # Try to repair it
    G, problem = check_everybody_is_attached(G, repair=False) # Check if the reparation worked.
    try:
        assert not problem
    except:
        print "I found some neighbouring sectors which do not have at least one couple of neighbouring navpoints."
        print "I could not fix it, I stop."
        raise

    # We check if, within each sectors, there is only one connected component
    G, problem = check_everybody_has_one_cc(G, repair=True)
    G, problem = check_everybody_has_one_cc(G, repair=False)
    try:
        assert not problem
    except:
        print "I found at least a sector which does not have only one connected component of napvoint."
        print "I could not fix it, I stop."
        raise

    if paras_G['attach_termination_nodes']:
        G = attach_termination_nodes(G)

    #draw_network_and_patches(None, G.G_nav, G.polygons, name = 'network3', show=False, flip_axes=True, dpi = 300, rep = rep)
    print


    ########## Generate Capacities and weights ###########
    print "Choosing capacities and weights..."
    G = give_capacities_and_weights(G, paras_G)
    print


    ############# Computing shortest paths ###########
    G.Nfp=paras_G['Nfp']
    G.G_nav.Nfp=G.Nfp
    
    print 'Computing shortest_paths (sectors) ...'
    pairs_deleted = G.compute_shortest_paths(G.Nfp, repetitions=False, old=False, delete_pairs=False)   
    G.infer_airports_from_short_list() # This could be removed in principle, because G.airports is not used anymore, only short.keys()

    pairs = G.G_nav.short.keys()[:]
    for (p1,p2) in pairs:
        s1=G.G_nav.node[p1]['sec']
        s2=G.G_nav.node[p2]['sec']
        if (s1, s2) in pairs_deleted:
            del G.G_nav.short[(p1,p2)]
            #G.G_nav.pairs.remove((p1,p2))
            print 'I removed the pair of navpoints', (p1,p2), 'because the corresponding pair of sectors', (s1, s2), 'has been removed.'

    G.G_nav.infer_airports_from_short_list() # Same remark

    #G=compute_sp_restricted(G, G.Nfp, silent=False)
     
    print 'Computing shortest_paths (navpoints) ...'
    print 'Number of pairs before computation:', len(G.G_nav.short.keys())  
    # TODO: resolve this problem of silence.
    G.compute_sp_restricted(G.Nfp, silent=False, delete_pairs=False)
    G.G_nav.infer_airports_from_short_list()
    G.check_repair_sector_airports()
    G.stamp_airports()
    G.G_nav.stamp_airports()

    print 'Number of pairs nav-entry/exit before checking flights:', len(G.G_nav.short.keys())
    #print 'Number of flights before checking flights:', len(paras_G["flights_selected"])
    G.check_airports_and_pairs() # No action here

    # if paras_G['flights_selected']!=None:
    #     flights_selected = G.check_all_real_flights_are_legitimate(paras_G['flights_selected'], repair=True)
    #     #G.check_all_real_flights_are_legitimate(paras_G['flights_selected'], repair=False)
    #     G.check_all_real_flights_are_legitimate(flights_selected, repair=False)

    #     # Give capacities and weights based on the new set of flights
    #     G = give_capacities_and_weights(G, paras_G)
    #flights_selected = paras_G["flights_selected"][:]
    print 'Selected finally', len(flights_selected), "flights."

        #G.check_all_real_flights_are_legitimate(flights_selected) # no action taken here

    print 'Final number of sectors:', len(G.nodes())
    print 'Final number of navpoints:', len(G.G_nav.nodes())
    print 'Final number of airports (sectors):', len(G.airports)
    print 'Final number of connections (sectors):', len(G.connections())
    print 'Final number of airports (navpoints):', len(G.G_nav.airports)
    print 'Final number of connections (navpoints):', len(G.G_nav.connections()) 

    ##################### Automatic Name #######################
    long_name = automatic_name(G, paras_G)
    

    ##################### Manual name #################
    if paras_G['name']!='': 
        name = paras_G['name']
    else:
        name = long_name #Automatic name is used only if the name is not specified otherwise
    
    G.name = name
    G.comments = {'long name':long_name, 'made with version':version}
    
    if save:
        if save_path==None: 
            save_path = join(rep, name)
        with open(save_path + '.pic','w') as f:
            pickle.dump(G, f)
        with open(save_path + '_flights_selected.pic','w') as f:
            pickle.dump(flights_selected, f)

    G.basic_statistics(rep = save_path + '_')

    print 'Network saved as', save_path + '.pic'
    #show_everything(G.polygons,G,save=True,name=name,show=False)       
    

    print 'Done.'
        
    if show:
        if paras_G['flights_selected']==None:
            draw_network_and_patches(None, G.G_nav, G.polygons, name=name, show=True, flip_axes=True, trajectories=G.G_nav.short.values(), rep = rep)
        else:
            trajectories = [[G.G_nav.idx_nodes[p[0]] for p in f['route_m1']] for f in paras_G['flights_selected']]
            draw_network_and_patches(None, G.G_nav, G.polygons, name=name, show=True, flip_axes=True, trajectories=trajectories, rep = rep)
    return G

if  __name__=='__main__':

    from paras_G import paras_G
    
    print 'Building hybrid network with paras_G:'
    print paras_G
    print

    rep = "../networks"
    G = prepare_hybrid_network(paras_G, rep = rep)