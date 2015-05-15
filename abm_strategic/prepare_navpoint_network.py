#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===========================================================================
This file is used to build a navpoint network with superimposed sector 
network. The main function is prepare_hybrid_network, with which the 
user can build a totally new network, or take some data from elsewhere.

TODO: write a Builder
===========================================================================
"""

import sys
sys.path.insert(1, '..')
import pickle
import networkx as nx
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union, unary_union
from random import choice, seed, sample
from string import split
from copy import deepcopy
from os.path import join

from simAirSpaceO import Net, NavpointNet
from utilities import clean_network, find_entry_exit

#Distance
from libs.tools_airports import get_paras, extract_flows_from_data, expand, dist_flat_kms
#Modules
from libs.general_tools import draw_network_and_patches, silence, counter, delay, date_st, make_union_interval, voronoi_finite_polygons_2d
from libs.paths import result_dir

version='2.9.9'

_colors = ('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')


def area(p):
    """
    Returns the area of a polygon.

    Parameters
    ----------
    p : list of tuples (x, y)
        Coordinates of the boundaries. Last Point must NOT be equal to first point.

    Returns
    -------
    Area based on cartesian distance
    """
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def attach_termination_nodes(G):
    """
    Attach navpoints with degree 1 to closest node in the same sector.
    TODO: infinite loop if a navpoint is alone in a sector (and with degree 1).

    Parameters
    ----------
    G : Hybrid network

    Returns
    -------
    G : Hybrid network
        where the lonely navpoints have been attached.

    Notes
    -----
    New in 2.9.6: 
    Changed in 2.9.7: do not attach airports.

    """

    changed = True
    while changed:
        changed = False
        for n in G.G_nav.nodes():
            sec = G.G_nav.node[n]['sec']
            if G.G_nav.degree(n)==1 and not n in G.G_nav.get_airports():
                changed = True
                # Find the nodes which are in the same sector, not already attached to n
                # and not on a border.
                pairs = [(n, n2) for n2 in G.node[sec]['navs'] if n2!=n and (not n2 in G.G_nav.neighbors(n)) and (not n in G.G_nav.navpoints_borders or not n2 in G.G_nav.navpoints_borders)]
                # Compute distance between potential attaching nodes and choose the closest one.
                distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                n1_selected, n2_selected = pairs[np.argmin(distances)]
                print "I attached node", n1_selected, "with degree 1 to node", n2_selected
                G.G_nav.add_edge(n1_selected, n2_selected)
    return G

def attach_two_sectors(s1, s2, G):
    """
    Attach two sectors by attaching two navpoints in each of them.
    The closest navpoints to each other are chosen.

    Parameters
    ----------
    s1 and s2 : indices of sectors
    G : Hybrid network

    Raises 
    ------
    AssertionError
        if there is no navpoint in s1 or s2

    Returns
    -------
    G : Hybrid network
        with the two sectors attached

    """

    navs_in_s1=G.node[s1]['navs']

    try:
        assert len(navs_in_s1)>0
    except AssertionError:
        print "There is no navpoint in sector", s1, "!"
        raise

    navs_in_s2=G.node[s2]['navs']
    try:
        assert len(navs_in_s2)>0
    except AssertionError:
        print "There is no navpoint in sector", s2, "!"
        raise
    
    # All possible connections between navpoints of both sectors
    pairs=[(n1, n2) for n1 in navs_in_s1 for n2 in navs_in_s2]

    # Compute distances of all possible pairs
    distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]

    # Select the closest navpoints
    n1_selected, n2_selected = pairs[np.argmin(distances)]
    G.G_nav.add_edge(n1_selected, n2_selected)

    return G

def automatic_name(G, paras_G):
    """
    Automatic name based on the parameters of construction.

    Parameters
    ----------
    G : finalized Net object.
    paras_G : dictionary
        The function extract some relevenant keys and values from it
        in order to build the name. 

    Returns
    -------
    long_name : string

    Notes
    -----
    Note all parameters (keys of paras_G) are taken into account.

    """

    long_name = G.type_of_net + '_N' + str(len(G.nodes()))
    
    if G.airports!=[] and len(G.airports)==2:
       long_name+='_airports' +  str(G.airports[0]) + '_' + str(G.airports[1])
    elif len(G.airports)>2:
        long_name+='_nairports' + str(len(G.airports))
    if paras_G['pairs_sec']!=[] and len(G.airports)==2:
        long_name+='_direction_' + str(paras_G['pairs_sec'][0][0]) + '_' + str(paras_G['pairs_sec'][0][1])
    long_name+='_cap_' + G.typ_capacities
    
    if G.typ_capacities!='manual':
        long_name+='_C' + str(paras_G['C'])
    long_name+='_w_' + G.typ_weights
    
    if G.typ_weights=='gauss':
        long_name+='sig' + str(paras_G['sigma'])
    long_name+='_Nfp' + str(G.Nfp)

    return long_name

def check_and_fix_empty_sectors(G, checked=None, repair=False):
    """
    Check-and-fix procedure for an hybrid network. 

    Parameters
    ----------
    G : Hybrid network 
        needs polygons of sectors as attributes.
    checked : dictionary, optional
        keys are ALL sectors and values are boolean, True if the sectors
        is not empty, False otherwise. If None, empty sectors are recomputed
    repair : bool, False
        if True, adds a cheaply computed point inside each empty sector.
        Updates the sectors' list of navpoints, then put some links between 
        this navpoint and the closest one off each neighboring sector.

    Returns
    -------
    G : modified hybrid network
    problem : bool
        True if at least one empty sector has been found during the procedure.

    Notes
    -----
    Possible improvements: put more than one points in the empty sectors.

    """

    problem = False
    try:
        if checked!=None:
            empty_sectors = [k for k,v in checked.items() if not v]
        else:
            empty_sectors = [n for n in G.nodes() if len(G.node[n]['navs'])==0]
        assert len(empty_sectors)==0
    except AssertionError:
        print "The following sectors do not have any navpoint left:", empty_sectors
        if repair:
            print "I add a navpoint at the centroid for them."
            print "I add also some links with the closest points of neighboring sectors."
            for sec in empty_sectors:
                nav = len(G.G_nav.nodes())
                RP = G.polygons[sec].representative_point()
                G.G_nav.add_node(nav, coord = RP.coords)

                if G.polygons[sec].contains(RP):
                    G.G_nav.node[nav]['sec']=sec
                    G.node[sec]['navs'].append(nav)
                else:
                    Exception("The representative point in not in the shape!")
                
                print 'G.neighbors(sec)', G.neighbors(sec)
                for sec2 in G.neighbors(sec):
                    pairs = []
                    for nav2 in G.node[sec2]['navs']:
                        pairs.append((nav, nav2))
                    distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                    n1_selected, n2_selected = pairs[np.argmin(distances)]
                    print "Adding edge between", n1_selected, "and", n2_selected
                    G.G_nav.add_edge(n1, n2)#, weight=np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])))
        problem = True
    except:
        raise

    return G, problem

def check_empty_polygon(G, repair = False):
    """
    Check all sector-polygons of G to see if they are reduced to a
    single point (in which case they have no representative_point).

    Parameters
    ----------
    G : networkx object with attribute polygons
        giving the dictionary of the shapely Polygons to check
    repair : boolean
        if set to True, the empty polygons are removed from the 
        dictionary

    """

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

    Parameters
    ----------
    G : Hybrid network
    repair : boolen, optional
        If True, attach the two closest navpoints in each sector.

    Returns
    -------
    G : hybrid network 
        untouched if repair=False
    problem : boolean
        True if a problem was detected (but it should have been fixed).

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
    Check that all sectors have only one connected component of navpoints.
    For each sector, the connected components are computed with the subnetwork 
    of G containing all navpoints in the sector.

    Parameters
    ----------
    G : Hybrid network
    repair : boolean, optional
        If True, attach the connected components by creating edges between 
        a node in each of them to a node in the biggest component. The pairs
        of nodes are chosen for each pair of components by using the closest 
        points.

    Returns
    -------
    G : hybrid network 
        untouched if repair=False
    problem : boolean
        True if a problem was detected (but it should have been fixed).

    Notes
    -----
    Remark: does not recompute weights for new edges.
    Changed in 2.9.6: we attach the nodes to the closest. Recursive check.
    Changed in 2.9.8: avoid infinite loop when there is no problem.

    """

    problem = False
    for s in G.nodes():
        nodes_in_s = [n for n in G.G_nav.nodes() if G.G_nav.node[n]['sec']==s]
        H_nav_s = G.G_nav.subgraph(nodes_in_s)
        cc = nx.connected_components(H_nav_s)
        problem_here = len(cc)>1
        while len(cc)>1 and repair:
            print 'Problem: sector', s, 'has more than one connected component (' + str(len(cc)) + ' exactly).'
            if repair:
                print "I'm fixing this."
                c1 = cc[0] # we attach everyone to the biggest cc.
                all_other_nodes = [n for c in cc for n in c if c!=c1]
                pairs = [(n1, n2) for n1 in c1 for n2 in all_other_nodes]
                distances = [np.linalg.norm(np.array(G.G_nav.node[n1]['coord']) - np.array(G.G_nav.node[n2]['coord'])) for n1, n2 in pairs]
                n1_selected, n2_selected = pairs[np.argmin(distances)]
                G.G_nav.add_edge(n1_selected,n2_selected)#, weight=w)
                H_nav_s.add_edge(n1_selected,n2_selected)#, weight=w)
            cc = nx.connected_components(H_nav_s)
        problem = problem or problem_here
    return G, problem

def check_matching(G, repair=False):
    """
    Check that the list of navpoints contained in each sector
    is consistent with the sector to which the napvoint belongs.

    Parameters
    ----------
    G : Hybrid network
    repair : boolean, optional
        If set to True, does two things:
        - add navpoint flagged as belonging to a sector to the list
        of navpoints in a sector
        - flag navpoint if the right sector if the found in the list 
        of navpoints of a sector.

    Returns
    -------
    G : hybrid network 
        untouched if repair=False
    problem : boolean
        True if a problem was detected (but it should have been fixed).

    Raises
    ------
    Exception : if a node lacks the key 'sec'. 

    Notes
    -----
    Changed in 2.9.8: do not raise Exception in case missing keys 'sec'
    for sectors

    """

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
                print "Navpoint", n, "have matches sector", G.G_nav.node[n]['sec'], \
                        "instead of sector", s, "."
                problem = True
                if repair:
                    print "I match", n, "to", s, "."
                    G.G_nav.node[n]['sec']=s

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

def compute_navpoints_borders(borders_coordinates, shape, lin_dens=10, only_outer_boundary=False,
    small=10**(-5.), thr=1e-2):
    """
    Put some navpoints on each segments given by borders. The points are shifted slightly to as to 
    belong to only one sector without any ambiguity.

    Parameters
    ----------
    borders_coordinates : list of segments
        Segments must 2-tuple of 2-tuples containing the coordinates.
    shape : shapely Polygon
        representing the union of all sectors. Only used to check if the 
        new navpoints are in the global shape (which should be the case).
    lin_dens : float
        Number of new navpoints per unit of length
    only_outer_boundary : bool
        if set to True, navpoints will be created only on the global_shape
        Otherwise the navpoints are created on every boundaries.
    thr : float 
        Spatial threshold for the detection of navpoint on the borders.
    small : float
        Used to shift the points.

    Returns
    -------
    navpoints : list of coordinates
        of the new points.

    Notes
    -----
    New in 2.9.7.
    Changed in 2.9.8: don't check if coordinates of the new points are wihtin a square of size 1x1
    """

    navpoints = []
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
            
            #if not shape.contains(P):
            #    print "Could not find any coordinates in the shape for this point."

            # if -1.<c[0]<1. and -1.<c[1]<1.:
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
        #print
    return navpoints

def compute_possible_outer_pairs(G):
    """
    Compute the legitimate pairs of entry/exits. 
    Legitimate means that entry/exit are in different non-neighbouring
    sectors.

    Parameters
    ----------
    G : hybrid network
        Must have attributes outer_nodes.

    Returns
    -------
    pairs : list of entry/exits 

    """

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

def compute_voronoi(G, xlims=(-1., 1.), ylims=(-1., 1.)):
    """
    Compute the voronoi tesselation of the network G. 
    Tessels are given to G through the polygons attribute,
    which is a list of shapely Polygons. In order to avoid 
    infinite polygons, the polygons are acute with a sqre of 
    size a.

    TODO: there is a problem with the outer polyongs which can be 
    GeomtryCollections (see corresponding unit test).

    Parameters
    ----------
    G : networkx object
        Each node of the network needs to have a 'coord' key
        with a 2-tuple for coordinates.
    a : float, optional
        size of the square which is used to cut the polygons.

    Returns
    -------
    G : same networkx object
        With a (new) attribute 'polygons', which is a list of shapely Polygons
        representing the cells.
    vor : Output of the Voronoi function of scipy.

    """
    polygons = {}
    nodes = G.nodes()

    # Compute the voronoi tesselation with scipy
    # print np.array([G.node[n]['coord'] for n in nodes])
    vor = Voronoi(np.array([G.node[n]['coord'] for n in nodes]))
    # print
    # print dir(vor)
    # print vor.point_region
    # print vor.vertices
    # print vor.ridge_points
    # print
    # print

    new_regions, new_vertices = voronoi_finite_polygons_2d(vor)
    # print new_vertices
    # print 
    # print new_regions

    voronoi_plot_2d(vor)
    #plt.show()

    # Build polygons objects
    #for i, p in enumerate(vor.point_region):
    #    r = vor.regions[p]
    #    coords=[vor.vertices[n] for n in r + [r[0]] if n!=-1]
    for i, p in enumerate(new_regions):
        coords = list(new_vertices[p]) + [new_vertices[p][0]]
        #print "YAYAH", p, coords
        if len(coords)>2:
            G.node[i]['area'] = area(coords)
            polygons[i] = Polygon(coords)
            try:
                assert abs(G.node[i]['area'] - polygons[i].area)<10**(-6.)
            except:
                raise Exception(i, G.node[i]['area'], polygons[i].area)


        else:
            # Case where the polygon is only made of 2 points...
            print "Problem: the sector", i, "has the following coords for its vertices:", coords
            print "I remove the corresponding node from the network."
            G.remove_node(nodes[i])
    
    # raise Exception()
    # eps = 0.1
    # minx, maxx, miny, maxy = min([n[0] for n in vor.vertices])-eps, max([n[0] for n in vor.vertices]) +eps, min([n[1] for n in vor.vertices]) -eps, max([n[1] for n in vor.vertices]) +eps
    
    square = Polygon([[xlims[0],ylims[0]], [xlims[0], ylims[1]], [xlims[1],ylims[1]], [xlims[1], ylims[0]], [xlims[0],ylims[0]]])

    # Cut all polygons with the square.
    for n, pol in polygons.items():
        polygons[n] = pol.intersection(square)
        
        try:
            assert type(polygons[n])==type(Polygon())
        except:
            print "BAM", n, 'coords:', coords
            raise

    G.polygons = polygons
    return G, vor

def detect_nodes_on_boundaries(paras_G, G, thr = 1e-2):
    """
    Detects the nodes on the OUTER boundary of the airspace.

    Parameters
    ----------
    paras_G : dictionary
        Must have the key make_borders_points with corresponding 
        boolean value. If False, the outer nodes are set to the 
        navpoints at borders. Otherwise they are recomputed.

    G : hybrid network
        Must have attributes global_shape and navpoints_borders. 
        Use the build method of G for that.

    thr : float, optional
        small number of geometrical detection.

    Returns
    -------
    G : modified hybrid network

    Notes
    -----
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
    Remove from the network sectors which have a number of navpoints
    smaller (or equal) than thr. Detection is based on the key 'navs' of navpoints.

    Parameters
    ----------
    G : hybrid network
    thr : int
        Number of navpoints under which the sector is removed

    Returns
    -------
    G : modified network
    removed : list
        of sectors removed.

    Notes
    -----
    New in 2.9.6: remove sectors having a small number of navpoints.
    Changed in 2.9.7: set thr<0 for no action (instead of 0).
    """

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
    Compute the average times needed for a filght to go from one node to the other.
    Segments of flights which are not edges of G are not considered.

    Parameters
    ----------
    G : Net or NavpointNet object 
        with matching dictionnary between names and indices of nodes as
        attribute 'idx_nodes'.
    flights : list of Flight Object from the Distance library
        Object needs to have a key 'route_m1t' which is a list of the navpoint label 
        and times (name, time), with time a tuple (y, m, d, h, m s).

    Returns
    -------
    weights : dictionnary
        keys are tuple (node index, node index) and values are the weights, 
        i.e. the average time needed to go from one node to the other, in minutes.

    Notes
    -----
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: flights are given externally.
    TODO: use datetime instead of delay.

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

def extract_capacity_from_traffic(G, flights, date=[2010, 5, 6, 0, 0, 0]):
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

def find_pairs(all_airports, all_pairs, nairports, G, remove_pairs_same_sector=False, n_tries=500):
    """
    Find nairports for which each of them have a link with at least one of the others (no
    isolated airport).

    The algorithm should never fail for nairport=2 (you can always find a pair) and of course 
    for naiports=len(all_airports) if there is no isolated airport in the original list. Return all pairs from all_pairs for which the origin and 
    destination are among the nairports. 
    
    Parameters
    ----------
    all_airports : list
        list of all airports among which the airports are drawn.
    all_pairs : list 2-tuple
        list of all pairs origins-destinations
    nairports : int
        Number of airports to return.
    remove_pairs_same_sector : boolean, optional
        If True, remove the pairs of airports which are in the same sector.

    Returns
    -------
    candidates : list
        of nairports airports taken among all_airports which have at least one link with 
        another airport in the same set.
    pairs : list of 2-tuple
        of pairs taken from all_pairs for which origin and destination are in candidates.

    Notes
    -----
    Changed in 2.9.3: Possibility of removing the pairs which have their airports in the same sector.
    Chnaged in 2.9.8: Completely modified, use connected components
    """
    
    if remove_pairs_same_sector:
        to_remove = []
        for p1, p2 in all_pairs:
            if G.G_nav.node[p1]['sec']==G.G_nav.node[p2]['sec']:
                to_remove.append((p1,p2))
        for p in to_remove:
            all_pairs.remove(p)

    # Build the network of airports
    net_air = nx.Graph()
    for a in all_airports:
        net_air.add_node(a)

    for a1, a2 in all_pairs:
        net_air.add_edge(a1, a2)

    #cc = [c for c in nx.connected_components(net_air) if len(c)>=nairports]

    # Brute force algorithm
    #while len(cc)>0:
        #c = cc[0]
    for n in range(n_tries):
        #candidates = sample(c, nairports)    
        candidates = sample(net_air.nodes(), nairports)    
        sub = net_air.subgraph(candidates)
        if min([len(c) for c in nx.connected_components(sub)])>1:
            pairs = sub.edges()
            return candidates, pairs
    #cc.pop(0)

    raise Exception('Impossible to find pairs with so many airports. Try with less airports.')



    # black_list = set()
    # candidates = sample(all_airports, nairports)
    # #found = len(all_airports)==nairports
    # found = False
    # #pairs = [p for p in all_pairs if p[0] in candidates and p[1] in candidates]
    # while not found and len(black_list)<(len(all_airports) - nairports):
    #     pairs = [p for p in all_pairs if p[0] in candidates and p[1] in candidates]
    #     selected = np.unique([p[0] for p in pairs] + [p[1] for p in pairs])
    #     found = True
    #     for candidate in candidates:
    #         if not candidate in selected: # isolated from the rest of the group
    #             candidates.remove(candidate)
    #             black_list.update([candidate])
    #             found = False
    #     if not found:
    #         candidates += sample([a for a in all_airports if not a in black_list], nairports - len(candidates))
    # if not found:
    #     raise Pouet('Impossible to find pairs with so many airports. Try with less airports.')
    # else:
    #     assert len(candidates)==nairports
    #     return candidates, pairs

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
        avg_length = np.mean([dist_flat_kms(np.array(G.G_nav.node[e[0]]['coord'])*60., np.array(G.G_nav.node[e[1]]['coord'])*60.) for e in G.G_nav.edges() if G.G_nav[e[0]][e[1]].has_key('weight')])
        for e in G.G_nav.edges():
            if not G.G_nav[e[0]][e[1]].has_key('weight'):
                #print G.G_nav.node[e[0]]['coord']
                #raise Exception()
                length = dist_flat_kms(np.array(G.G_nav.node[e[0]]['coord'])*60., np.array(G.G_nav.node[e[1]]['coord'])*60.)
                weight = avg_weight*length/avg_length
                print "This edge did not receive any weight:", e, ", I set it to the average (", weight, "minutes)"
                G.G_nav[e[0]][e[1]]['weight'] = weight
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

def navpoints_at_borders(G, lin_dens=10., only_outer_boundary=False, thr=1e-2):
    """
    Create navpoints at the boundaries of all sectors or only on the boundary
    of the aggregated shape. Build the list of borders and then call 
    compute_navpoints_borders.

    Parameters
    ----------
    G : networkx object with list of polygons and global_shape
        global_shape must be the union of all polygons.
    lin_dens : float
        number of points per unit of length on the border
    only_outer_boundary : bool
        if set to True, navpoints will be created only on the global_shape
        Otherwise the navpoints are created on every boundaries.
    thr : float 
        Spatial threshold for the detection of navpoint on the borders.

    Returns
    -------
    navpoints : list of coordinates of new navpoints.

    Notes
    -----
    Changed in 2.9.7: don't use vor anymore. More efficient.

    """
    shape = G.global_shape

    # Build the list of borders
    if only_outer_boundary:
        # Only the exterior of the global_shape
        borders_coords = list(shape.exterior.coords)
        borders = [(borders_coords[i], borders_coords[i+1]) for i in range(len(borders_coords)-1)]
    else:
        # All borders of all sectors
        borders = []
        for n, pol in G.polygons.items():
            for i in range(len(pol.exterior.coords)-1):
                seg = (pol.exterior.coords[i], pol.exterior.coords[i+1])
                if not (seg in borders or seg[::-1] in borders): # this is to ensure non-redundancy of the borders.
                    borders.append(seg)

    navpoints = compute_navpoints_borders(borders, shape, lin_dens=lin_dens, only_outer_boundary=only_outer_boundary, thr=thr)

    print "I will create", len(navpoints), "navpoints on the borders."
    return navpoints

def prepare_sectors_network(paras_G):  
    """
    Prepare the network of sectors which will be the base for the navpoint network. If no file 
    containing a networkx object is given via paras_G['net_sec'], it builds a new sector network
    with paras_G['N'] number of sectors, including one sector at each corner of the square 
    [0, 1] x [0, 1]. The latter is done to facilitate the voronoi tesselation and the subsequent
    computation of the boundaries of the sectors.

    Parameters
    ----------
    paras_G : dictionary
        of the parameters for the construction of the network

    Returns
    -------
    G : Net object.

    Notes
    -----
    New in 2.9.6: refactorization.
    Changed in 2.9.8: updating signature of build function
    Changed in 2.9.9: removed the possiblity of adding airports.

    """

    G = Net()
    G.type = 'sec' #for sectors
    G.type_of_net = paras_G['type_of_net']

    # Import the network from the paras if given, build a new one otherwise
    if paras_G['net_sec']!=None:
        G.import_from(paras_G['net_sec'], numberize=((type(paras_G['net_sec'].nodes()[0])!=type(1.)) and (type(paras_G['net_sec'].nodes()[0])!=type(1))))
    else:
        #G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'],put_nodes_at_corners = True)
        if paras_G['N']<=4:
            print "WARNING: you requested less than 4 sectors, so I do not put any sectors on the corners of the square."
            paras_G['N']+=4
        G.build(paras_G['N']-4, Gtype=paras_G['type_of_net'], put_nodes_at_corners=paras_G['N']>4)

    # Give the pre-built polygons to the network.
    if paras_G['polygons']!=None:
        G.polygons={}
        for name, shape in paras_G['polygons'].items():
            G.polygons[G.idx_nodes[name]]=shape
    else:
        G, vor = compute_voronoi(G)
    
    # Check if every sector has a polygon
    for n in G.nodes():
        try:
            assert G.polygons.has_key(n)
        except:
            print "Sector", n, "doesn't have any polygon associated!"
            raise

    # Check if there are polygons reduced to a single point.
    check_empty_polygon(G, repair=True)
    check_empty_polygon(G, repair=False)

    # Make sure that neighbors have a common boundary
    recompute_neighbors(G)

    # Convex hull of the sectors
    G.global_shape = unary_union(G.polygons.values()).convex_hull

    # if not no_airports:
    #     if paras_G['airports_sec']!=[]:
    #         G.add_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
    #     else:
    #         G.generate_airports(paras_G['nairports'],paras_G['min_dis'])

    return G
    
def recompute_neighbors(G):
    """
    Checks if neighbouring sectors have a common boundary. Disconnects them otherwise.

    Parameters
    ----------
    G : networkx object with attribute polygons
        'polygons' must be a list of shapely Polygons.

    Notes
    -----
    New in 2.9.6
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
    """
    Remove all nodes in airports list and list of origin-destination 
    which are not in the list of nodes of G. 
    
    Parameters
    ----------
    G : networkx object
    pairs : list of 2-tuples (int, int)
        list of origin-destination pairs. If set to None, the procedure
        is skipped for this list.
    airports : list of integers
        list of possible airports. If set to None, the procedure is 
        skipped for this list.

    Returns
    -------
    pairs : same as input without the tuples including nodes absent of G
    airports : same as input without nodes absent of G

    """

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
    """
    Remove from pairs_nav the ones for which both navpoints are
    in the same sector.

    Parameters
    ----------
    G : hybrid network
    pairs_nav : list of 2-tuple
        list of pairs of nodes of navpoints.

    Returns
    -------
    pairs_nav : list of 2-tuple
        same as input with pairs in the same sector removed.

    """

    for n1, n2 in pairs_nav[:]:
        if G.G_nav.node[n1]['sec'] == G.G_nav.node[n2]['sec']:
            pairs_nav.remove((n1, n2))

    return pairs_nav

def segments(p):
    """
    Compute the list of segments given a list of coordinates. Attach
    the end point to the first one.

    Parameters
    ----------
    p : list of tuples (x, y)
        Coordinates of the boundaries.

    Returns
    l : list of 2-tuples of 2-tuples.
    """

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

def prepare_hybrid_network(paras_G, rep=None, save_name=None, show=True):
    """
    This is a builder for an hybrid network. An ``hybrid network'' is a Net object (the class is 
    defined in the file simAirSpaceO.py) which has an attribute which is a NavpointNet object 
    (also defined in simAirSpaceO.py). The Net object, let us call it G, is the support for 
    all the information concerning the sectors, whereas the NavpointNet object -- $G_{nav}$ -- 
    includes all information related to the navigation point network. The latter can be accessed 
    using $G.G_{nav}$.

    The builder is auite versatile and can be used with some ata concerning the airspace or can
    take some features from traffic data, like the entry/exits, the capacities, etc.

    Parameters
    ----------
    paras_G : dictionnary
         include all the relevant parameters to apply. Such a dictionnary can be
         generated with the paras_G.py file.
    rep : string, optional
        full path of the directory where to save the network and related files (like statistics).
        If None, nothing is saved
    save_name :  string, optional
        If not None, specifies the name of the file where the network is saved. If None, the name
        of the network given by paras_G['name'] will be used.
    show : boolean, optional
        if True, display an image of the network.

    Returns
    -------
    G : hybrid network.
        Hybrid networks have an attribute G_nav which is the navpoint network.

    Notes
    -----
    New in 2.9.6: refactorization.
    Changed in 2.9.7: - navpoints are not contained in any sectors but touch one are linked to it.
                      - numberize also airports and pairs.
    Changed in 2.9.8: supports single sector network TODO. Supports custom external function 
        for choosing airports.
    TODO: make a builder class with methods.

    """
    
    print

    ############ Prepare network of sectors #############
    print "Preparing network of sector..."
    G = prepare_sectors_network(paras_G)
    if len(G.edges())==0:
        raise NoEdges("The sector network has only one sector, this feature is not supported for now.")#TODO
    #G.show()
    print


    ############ Prepare nodes and edges of network of navpoints ###########
    print "Preparing nodes and edges of network of navpoints..."
    for n in G.nodes():
        G.node[n]['navs']=[]
        
    G.G_nav = NavpointNet()
    G.G_nav.type = 'nav'

    if paras_G['net_nav']==None:
        nav_at_borders = navpoints_at_borders(G, only_outer_boundary=not paras_G['make_borders_points'], lin_dens=paras_G['lin_dens_borders'])
        #nav_at_borders = []
        # Remark: paras_G['nairports'] is not used here because of the generation_of_airports=False.
        G.G_nav.build((paras_G['N_by_sectors']-1)*len(G.nodes()), 0, paras_G['min_dis'], generation_of_airports=False, \
                sector_list=[G.node[n]['coord'] for n in G.nodes()], navpoints_borders=nav_at_borders, shortcut=0.01)

        idx = max(G.G_nav.nodes())
    else:
        numberize = ((type(paras_G['net_nav'].nodes()[0])!=type(1.)) and (type(paras_G['net_nav'].nodes()[0])!=type(1)))
        G.G_nav.import_from(paras_G['net_nav'], numberize=numberize) 
        idx = max(G.G_nav.nodes())
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

    if paras_G['airports_nav']!=None:
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
                print "Removed", nav, "from list of airports."
                paras_G['airports_nav'].remove(nav)
            # Remove node from list of pairs
            for ee in paras_G['pairs_nav'][:]:
                if nav in set(ee):
                    print "Removed", ee, "from list of pairs."
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
        if paras_G['airports_nav']!=None:
            for n in paras_G['airports_nav']:
                assert n in G.G_nav.nodes()
    except AssertionError:
        raise Exception("Navpoint", n, "is not in the list of nodes of G.G_nav")
    #draw_network_and_patches(None, G.G_nav, G.polygons, show=True, flip_axes=True, rep = rep)


    ########### Choose the airports #############
    # `Airports' means all entry and exit points here, not only physical airports.
    print "Choosing the airports..."

    paras_G['pairs_nav'], paras_G['airports_nav'] = reduce_airports_to_existing_nodes(G.G_nav, paras_G['pairs_nav'], paras_G['airports_nav'])
    paras_G['pairs_sec'], paras_G['airports_sec'] = reduce_airports_to_existing_nodes(G, paras_G['pairs_sec'], paras_G['airports_sec'])

    if not paras_G['use_only_outer_nodes_for_airports']: 
        if paras_G['airports_sec']!=None and paras_G['airports_nav']!=None:
            # TODO
            print "WARNING: specifiying airports for both sectors and navpoints does make sense unless they are consitent with each other."
            G.add_airports(paras_G['airports_sec'], -10, pairs=paras_G['pairs_sec'], C_airport=paras_G['C_airport'])
        elif paras_G['airports_sec']!=None and paras_G['airports_nav']==None:
            # If the airports_sec are specified but not the airports_nav, add the former to network and then infer the latter.
            G.add_airports(paras_G['airports_sec'], -10, pairs=paras_G['pairs_sec'], C_airport=paras_G['C_airport'])
            G.G_nav.infer_airports_from_sectors(G.airports, paras_G['min_dis'])
        elif paras_G['airports_sec']==None and paras_G['airports_nav']!=None:
            # The other way round.
            if paras_G['function_airports_nav']!=None:
                # Custom function of airport building based on traffic.
                print "Extracting airports and pairs with custom function..."
                assert 'flights_selected' in paras_G.keys() and paras_G['flights_selected']!=None
                paras_G['flights_selected'], paras_G['airports_nav'], paras_G['pairs_nav'] = paras_G['function_airports_nav'](G.G_nav, paras_G['flights_selected'])
                print "Deleted flights because no navpoint of the trajectories belonged to the network:", len(paras_G['flights_selected']) - len(flights_selected)
            
            if not paras_G['singletons'] and paras_G['pairs_nav']!=None: 
                # Remove the pairs of navpoints which yield the same sec-entry and sec-exit.
                paras_G['pairs_nav'] = remove_singletons(G, paras_G['pairs_nav'])

            # Add airports to the nav-net and infer the airports for the sec-net.
            G.G_nav.add_airports(paras_G['airports_nav'], paras_G['min_dis'], pairs= paras_G['pairs_nav'], C_airport=paras_G['C_airport'])
            G.infer_airports_from_navpoints(paras_G['C_airport'], singletons=paras_G['singletons'])
        else:
            # If none of them are specified, draw at random some entry/exits for the navpoint network and
            # infer the sector airports.
            G.G_nav.generate_airports(paras_G['nairports_nav'], paras_G['min_dis'], C_airport=100000)
            G.infer_airports_from_navpoints(paras_G['C_airport'], singletons=paras_G['singletons'])
    else:
        G = detect_nodes_on_boundaries(paras_G, G)
        pairs_outer = compute_possible_outer_pairs(G)
        print "Found outer nodes:", G.G_nav.outer_nodes
        print "With", len(pairs_outer), "corresponding pairs."
        print "I add them as airports."
        G.G_nav.add_airports(G.G_nav.outer_nodes, paras_G['min_dis'], pairs=pairs_outer)
        G.infer_airports_from_navpoints(paras_G['C_airport'], singletons=paras_G['singletons'])

    print 'Number of airports (sectors) at this point:', len(G.airports)
    print 'Number of connections (sectors) at this point:', len(G.connections())
    print 'Number of airports (navpoints) at this point:', len(G.G_nav.airports)
    print 'Number of connections (navpoints) at this point:', len(G.G_nav.connections()) 

    try:
        if  paras_G['flights_selected']!=None:
            for f in paras_G['flights_selected']:
                e1, e2 = find_entry_exit(G.G_nav, f)
                assert e1, e2 in G.G_nav.connections()
    except AssertionError:
        raise Exception("e1, e2", e1, e2)

    #assert 333 in G.G_nav.nodes()
    #assert (333, 197) in G.G_nav.connections()

    ############# Repair some stuff #############
    print "Repairing mutiple issues..."
    
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
    G.Nfp = paras_G['Nfp']
    G.G_nav.Nfp = G.Nfp
    
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
    #flights_selected = paras_G["flights_selected"][:]0

    # Check distribution of velocities
    velocities = {(n1, n2):dist_flat_kms(np.array(G.G_nav.node[n1]['coord'])*60., np.array(G.G_nav.node[n2]['coord'])*60.)/(G.G_nav[n1][n2]['weight']/60.) for n1, n2 in G.G_nav.edges()}
    for e, vel in velocities.items():
        if vel>1200:
            print "edge", e , "has velocity:", vel
    #plt.hist(velocities.values(), bins = 100)
    #plt.show()

    if paras_G['flights_selected']!=None:
        print 'Selected finally', len(paras_G['flights_selected']), "flights."

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
    
    if save_name==None:
        save_name = name

    if rep!=None:
        with open(join(rep, save_name) + '.pic','w') as f:
            pickle.dump(G, f)
        if paras_G['flights_selected']!=None:
            with open(join(rep, save_name + '_flights_selected.pic'),'w') as f:
                pickle.dump(flights_selected, f)

    G.basic_statistics(rep=rep, name=save_name)

    print 'Network saved as', join(rep, save_name)+'.pic'
    #show_everything(G.polygons,G,save=True,name=name,show=False)       
    

    print 'Done.'
        
    if show:
        if paras_G['flights_selected']==None:
            draw_network_and_patches(None, G.G_nav, G.polygons, name=save_name, show=True, flip_axes=True, trajectories=[sp for paths in G.G_nav.short.values() for sp in paths], rep=rep)
        else:
            trajectories = [[G.G_nav.idx_nodes[p[0]] for p in f['route_m1']] for f in paras_G['flights_selected']]
            draw_network_and_patches(None, G.G_nav, G.polygons, name=save_name, show=True, flip_axes=True, trajectories=trajectories, rep=rep)
    return G

if  __name__=='__main__':

    if 1:
        # Manual seed
        see_ = 2
        #see_ = 15 #==> the one I used for new, old_good, etc.. 
        # see=2
        print "===================================="
        print "USING SEED", see_
        print "===================================="
        seed(see_)

    from paras_G import paras_G
    
    print 'Building hybrid network with paras_G:'
    print paras_G
    print

    rep = join(result_dir, "networks")
    #G = prepare_hybrid_network(paras_G, rep=rep)
    G = prepare_hybrid_network(paras_G, rep=rep)