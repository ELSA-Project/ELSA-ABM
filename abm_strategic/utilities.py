# -*- coding: utf-8 -*-
"""
Utilies for the ABM.
"""
import sys
sys.path.insert(1, '..')
import os
from mpl_toolkits.basemap import Basemap
from math import sqrt, cos, sin, pi, atan2, ceil
import numpy as np
import matplotlib.gridspec as gridspec
from descartes import PolygonPatch
import matplotlib.pyplot as plt
#from random import shuffle, seed
import networkx as nx
import imp
import pickle
from os.path import join
from string import split
from random import choice
from copy import deepcopy

from libs.general_tools import  delay, date_human, date_st, flip_polygon
from libs.tools_airports import bet_OD
from libs.efficiency import rectificate_trajectories_network

version='2.9.1'

#seed(3)
_colors=['Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed','Red']

#shuffle(_colors)

# ============================================================================ #
# =============================== Plotting =================================== #
# ============================================================================ #

def draw_network_map(G_init, title='Network map', trajectories=[], rep='./', airports=True, 
    load=False, generated=False, add_to_title='', polygons=[], numbers=False, show=True,
    colors='b', figsize=(9, 6), flip_axes=False, weight_scale=4., sizes=20.):
    """
    Utility used to plot a network and possibly trajectories, sectors, etc.

    """

    print "Drawing network..."
    G = deepcopy(G_init)
    polygons_copy = deepcopy(polygons)
    if flip_axes:
        for n in G.nodes():
            G.node[n]['coord']=(G.node[n]['coord'][1], G.node[n]['coord'][0])
        if polygons_copy!=[]:
            polygons_copy=[flip_polygon(pol) for pol in polygons_copy]

    nodes = G.nodes()[:]
    if type(colors)!=type('n'):
        colors = [colors[n] for n in nodes]
    if type(sizes)!=type(20.):
        sizes = [sizes[n] for n in nodes]

    x_min=min([G.node[n]['coord'][0]/60. for n in G.nodes()])-0.5
    x_max=max([G.node[n]['coord'][0]/60. for n in G.nodes()])+0.5
    y_min=min([G.node[n]['coord'][1]/60. for n in G.nodes()])-0.5
    y_max=max([G.node[n]['coord'][1]/60. for n in G.nodes()])+0.5
    

    #(x_min,y_min,x_max,y_max),G,airports,max_wei,zone_geo = rest
    fig=plt.figure(figsize=figsize)#*(y_max-y_min)/(x_max-x_min)))#,dpi=600)
    #gs = gridspec.GridSpec(1, 2, width_ratios=[6.,1.])
    gs = gridspec.GridSpec(1, 2, width_ratios=[6.,1.])
    ax = plt.subplot(gs[0])
    #ax.set_aspect(1./0.8)
    ax.set_aspect(figsize[0]/float(figsize[1]))

    
    if generated:
        def m(a,b):
            return a,b
        y,x=[G.node[n]['coord'][0] for n in nodes], [G.node[n]['coord'][1] for n in nodes]
    else:
        m=draw_zonemap(x_min,y_min,x_max,y_max,'i')
        x,y=split_coords(G, nodes, r=0.08)
    
    for i,pol in enumerate(polygons_copy):
        patch = PolygonPatch(pol,alpha=0.5, zorder=2, color=_colors[i%len(_colors)])
        ax.add_patch(patch) 

    if load:
        sze=[(np.average([G.node[n]['load'][i][1] for i in range(len(G.node[n]['load'])-1)],\
        weights=[(G.node[n]['load'][i+1][0] - G.node[n]['load'][i][0]) for i in range(len(G.node[n]['load'])-1)])
        /float(G.node[n]['capacity'])*800 + 5) for n in nodes]
    else:
        sze=sizes
        
    coords={n:m(y[i],x[i]) for i,n in enumerate(nodes)}
    
    ax.set_title(title)
    sca=ax.scatter([coords[n][0] for n in nodes],[coords[n][1] for n in nodes], marker='o', zorder=6, s=sze, c=colors)#,s=snf,lw=0,c=[0.,0.45,0.,1])
    if airports:
        scairports=ax.scatter([coords[n][0] for n in G.airports],[coords[n][1] for n in G.airports],marker='o', zorder=6, s=20., c='r')#,s=snf,lw=0,c=[0.,0.45,0.,1])

    if 1:
        for e in G.edges():
            plt.plot([coords[e[0]][0],coords[e[1]][0]],[coords[e[0]][1],coords[e[1]][1]],'k-',lw=0.5)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
          
    #weights={n:{v:0. for v in G.neighbors(n)} for n in G.nodes()}
    weights={n:{} for n in nodes}
    for path in trajectories:
        try:
            #path=f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p
            for i in range(0,len(path)-1):
                #print path[i], path[i+1]
                #weights[path[i]][path[i+1]]+=1.
                weights[path[i]][path[i+1]] = weights[path[i]].get(path[i+1], 0.) + 1.
        except ValueError: # Why?
            pass
        except:
            print "weights[path[i]]:", weights[path[i]]
            raise
    
    max_w=np.max([w for vois in weights.values() for w in vois.values()])
     
    for n,vois in weights.items():
        for v,w in vois.items():
           # if G.node[n]['m1'] and G.node[v]['m1']:
                plt.plot([coords[n][0],coords[v][0]],[coords[n][1],coords[v][1]],'r-',lw=w/max_w*weight_scale)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)

    if numbers:
        for n in G.nodes():
            plt.text(G.node[n]['coord'][0], G.node[n]['coord'][1], ster(n))
       # if 0:
       #     patch=PolygonPatch(adapt_shape_to_map(zone_geo,m),facecolor='grey', edgecolor='grey', alpha=0.08,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
       #     ax.add_patch(patch)
           
       # if 0:
       #     patch=PolygonPatch(adapt_shape_to_map(expand(zone_geo,0.005),m),facecolor='brown', edgecolor='black', alpha=0.1,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
       #     ax.add_patch(patch)
    plt.savefig(rep + 'network_flights' + add_to_title + '.png',dpi=300)
    if show:
        plt.show()

def find_entry_exit(G_nav, f, names=False):
    """
    Returns the first nodes (forward/backward) of flight f which belongs to G_nav.
    If no node of f are in G_nav, returns fisrt and last point (TODO: change this?)

    Parameters
    ----------
    G_nav : NavpointNet object
        must have attribute idx_nodes for mapping between index of nodes and real
        labels used in f.
    f : dictionary
        Possibly coming from Distance library. Needs to have a key 'route_m1t' 
        which is a list of tuples (label, time).
    names : boolean, optional
        If True, returns the labels. Otherwise, returns indices.
    
    """
    # Find the first node in trajectory which is in G_nav
    idx_entry = 0
    while idx_entry<len(f['route_m1t']) and not G_nav.idx_nodes[f['route_m1t'][idx_entry][0]] in G_nav.nodes():
        idx_entry += 1
    if idx_entry==len(f['route_m1t']): idx_entry = 0
    
    # Find the first node in trajectory which is in G_nav (backwards).
    idx_exit = -1
    while abs(idx_exit)<len(f['route_m1t']) and not G_nav.idx_nodes[f['route_m1t'][idx_exit][0]] in G_nav.nodes():
        idx_exit -= 1
    if idx_exit==len(f['route_m1t']): idx_exit = -1

    if names:
        return f['route_m1t'][idx_entry][0], f['route_m1t'][idx_exit][0]
    else:
        _entry = G_nav.idx_nodes[f['route_m1t'][idx_entry][0]]
        _exit = G_nav.idx_nodes[f['route_m1t'][idx_exit][0]]

        return _entry, _exit

def split_coords(G, nodes, r=0.04):
    """
    This is used for plotting map. It detects nodes which are too close
    to each other in the horizontal plane and move them slightly in a
    circle of radius r. 

    Parameters
    ----------
    G : networkx object
        nodes must have key 'coord' with correponding value (lat, long)
        in minutes degree.
    nodes : list
        of labels of nodes to check.
    r : float, optional
        radius (in unit 'degree')

    Returns
    -------
    x, y : lists of float
        representing latitudes and longitudes in degree. Coordinates are 
        not modified for nodes which are not too close to each other.
    
    """
    
    lines = []
    for n in G.nodes():
        if n in nodes:
            added = False
            for l in lines:
                if sqrt((G.node[n]['coord'][0] - G.node[l[0]]['coord'][0])**2 + (G.node[n]['coord'][1] - G.node[l[0]]['coord'][1])**2)<1.: #nodes closer than 0.1 degree
                    l.append(n)
                    added = True
            if not added:
                lines.append([n])
    
    for l in lines[:]:
        if len(l)==1:
            lines.remove(l)

    pouet = {}
    for l in lines:
        for n in l:
            pouet[n] = l
    x, y = [], []
    for n in nodes:
        if not n in pouet.keys():
            x.append(G.node[n]['coord'][0]/60.)
            y.append(G.node[n]['coord'][1]/60.)
        else:
            l = pouet[n]
            theta = 2.*pi*float(l.index(n))/float(len(l))
            x.append(G.node[n]['coord'][0]/60. + r*cos(theta))
            y.append(G.node[n]['coord'][1]/60. + r*sin(theta))
    return x, y
    
def draw_zonemap(x_min, y_min, x_max, y_max, res):
    """
    Wrapper to use basemap easily.

    Parameters
    ----------
    x_min, y_min, x_max, y_max : floats
        min and max latitudes and longitudes of the map in degree
    res : string
        resolution of the map. Use 'i' for medium resolution.

    Returns
    -------
    m : Basemap object
        which can be used for converting coordinates in the projection chosen 
        (gall-peters.)

    """ 

    m = Basemap(projection='gall', lon_0=0., llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max, resolution=res)
    m.drawmapboundary(fill_color='white') #set a background colour
    m.fillcontinents(color='white', lake_color='white')  # #85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=0.8)
    m.drawcountries(color='#6D5F47', linewidth=0.8)
    m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb')
    m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')
    return m

# ============================================================================ #
# =============================== Parameters ================================= #
# ============================================================================ #

class Paras(dict):
    """
    Class Paras
    ===========
    Custom dictionnary used to update parameters in a controlled way.
    This class is useful in case of multiple iterations of simulations
    with sweeping parameters and more or less complex interdependances
    between variables.
    In case of simple utilisation with a single iteration or no sweeping,
    a simple dictionary is enough.

    The update process is based on the attribute 'update_priority', 'to_update'.

    The first one is a list of keys. First entries should be updated before updating 
    later ones.

    The second is a dictionary. Each value is a tuple (f, args) where f is function
    and args is a list of keys that the function takes as arguments. The function
    returns the value of the corresponding key. 

    Notes
    -----
    'update_priority' and 'to_update' could be merged in an sorted dictionary.

    """
    
    def __init__(self, dic):
        for k,v in dic.items():
            self[k]=v
        self.to_update={}

    def update(self, name_para, new_value):
        """
        Updates the value with key name_para to new_value.

        Parameters
        ----------
        name_para : string
            label of the parameter to be updated
        new_value : object
            new value of entry name_para of the dictionary.

        Notes
        -----
        Changed in 2.9.4: self.update_priority instead of update_priority.

        """
        
        self[name_para] = new_value
        # Everything before level_of_priority_required should not be updated, given the para being updated.
        lvl = self.levels.get(name_para, len(self.update_priority)) #level_of_priority_required
        #print name_para, 'being updated'
        #print 'level of priority:', lvl, (lvl==len(update_priority))*'(no update)'
        for j in range(lvl, len(self.update_priority)):
            k = self.update_priority[j]
            (f, args) = self.to_update[k]
            vals = [self[a] for a in args] 
            self[k] = f(*vals)

    def analyse_dependance(self):
        """
        Detect the first level of priority hit by a dependance in each parameter.
        Those who don't need any kind of update are not in the dictionnary.

        This should be used once when the 'update_priority' and 'to_update' are 
        finished.

        It computes the attribute 'levels', which is a dictionnary, whose values are 
        the parameters. The values are indices relative to update_priority at which 
        the update should begin when the parameter corresponding to key is changed. 

        """

        # print 'Analysing dependances of the parameter with priorities', self.update_priority
        self.levels = {}
        for i, k in enumerate(self.update_priority):
            (f, args) = self.to_update[k]
            for arg in args:
                if arg not in self.levels.keys():
                    self.levels[arg] = i
 
def read_paras(paras_file=None, post_process=True):
    """
    Reads parameter file for a single simulation.
    """
    if paras_file==None:
        import my_paras as paras_mod
    else:
        paras_mod = imp.load_source("paras", paras_file)
    paras = paras_mod.paras

    if post_process:
        paras = post_process_paras(paras)

    return paras

def read_paras_iter(paras_file=None):
    """
    Reads parameter file for a iterated simulations.
    """
    if paras_file==None:
        import my_paras_iter as paras_mod
    else:
        paras_mod = imp.load_source("paras_iter", paras_file)
    paras = paras_mod.paras

    return paras

def post_process_paras(paras):
    ##################################################################################
    ################################# Post processing ################################
    ##################################################################################
    # This is useful in case of change of parameters (in particular using iter_sim) in
    # order to record the dependencies between variables.
    update_priority=[]
    to_update={}

    # -------------------- Post-processing -------------------- #

    paras['par']=tuple([tuple([float(_v) for _v in _p])  for _p in paras['par']]) # This is to ensure hashable type for keys.

    # Load network
    if paras['file_net']!=None:
        with open(paras['file_net']) as f:
            paras['G'] = pickle.load(f)
    
    if not 'G' in paras.keys():
        paras['G'] = None

    if paras['file_traffic']!=None:
        with open(paras['file_traffic'], 'r') as _f:
            flights = pickle.load(_f)
        paras['traffic'] = flights
        paras['flows'] = {}
        for f in flights:
            # _entry = G.G_nav.idx_navs[f['route_m1t'][0][0]]
            # _exit = G.G_nav.idx_navs[f['route_m1t'][-1][0]]
            if paras['G']!=None: 
                # # Find the first node in trajectory which is in airports
                # idx_entry = 0
                # while idx_entry<len(f['route_m1t']) and not paras['G'].G_nav.idx_nodes[f['route_m1t'][idx_entry][0]]:# in paras['G'].G_nav.airports:
                #     idx_entry += 1
                # if idx_entry==len(f['route_m1t']): idx_entry = 0
                
                # # Find the first node in trajectory which is in airports (backwards).
                # idx_exit = -1
                # while abs(idx_exit)<len(f['route_m1t']) and not paras['G'].G_nav.idx_nodes[f['route_m1t'][idx_exit][0]]:# in paras['G'].G_nav.airports:
                #     idx_exit -= 1
                # if idx_exit==len(f['route_m1t']): idx_exit = -1

                _entry, _exit = find_entry_exit(paras['G'].G_nav, f, names=True)
            else:
                idx_entry = 0
                idx_exit = -1
                _entry = f['route_m1t'][idx_entry][0]
                _exit = f['route_m1t'][idx_exit][0]

            paras['flows'][(_entry, _exit)] = paras['flows'].get((_entry, _exit),[]) + [f['route_m1t'][0][1]]

        if not paras['bootstrap_mode']:
            #paras['departure_times'] = 'exterior'
            paras['ACtot'] = sum([len(v) for v in paras['flows'].values()])
            paras['control_density'] = False
        else:
            if not 'ACtot' in paras.keys():
                paras['ACtot'] = sum([len(v) for v in paras['flows'].values()])
           
        #print 'pouet' 
        #print paras['ACtot']
        density=_func_density_vs_ACtot_na_day(paras['ACtot'], paras['na'], paras['day'])

        # There is no update requisites here, because the traffic should not be changed
        # when it is extracted from data.

    else:
        paras['flows'] = {}
        paras['times']=[]
        if paras['file_times'] != None:
            if paras['departure_times']=='from_data': #TODO
                with open('times_2010_5_6.pic', 'r') as f:
                    paras['times']=pickle.load(f)
        else:
            if paras['control_density']:
                # ACtot is not an independent variable and is computed thanks to density
                paras['ACtot']=_func_ACtot_vs_density_day_na(paras['density'], paras['day'], paras['na'])
                to_update['ACtot']=(_func_ACtot_vs_density_day_na, ('density', 'day', 'na'))
            else:
                # Density is not an independent variables and is computed thanks to ACtot.
                paras['density']=_func_density_vs_ACtot_na_day(paras['ACtot'], paras['na'], paras['day'])
                to_update['density']=(_func_density_vs_ACtot_na_day,('ACtot','na','day'))

            assert paras['departure_times'] in ['zeros','from_data','uniform','square_waves']

            if paras['departure_times']=='square_waves':
                paras['Np'] = _func_Np(paras['day'], paras['width_peak'], paras['Delta_t'])
                to_update['Np']=(_func_Np,('day', 'width_peak', 'Delta_t'))
                update_priority.append('Np')

                if paras['control_ACsperwave']:
                    # density/ACtot based on ACsperwave
                    paras['density'] = _func_density_vs_ACsperwave_Np_na_day(paras['ACsperwave'], paras['Np'], paras['ACtot'], paras['na'], paras['day'])
                    to_update['density']=(_func_density_vs_ACsperwave_Np_na_day,('ACsperwave', 'Np', 'ACtot', 'na', 'day'))
                    update_priority.append('density')   
                else:
                    # ACperwave based on density/ACtot
                    paras['ACsperwave']=_func_ACsperwave_vs_density_day_Np(paras['density'], paras['day'], paras['Np'])
                    to_update['ACsperwave']=(_func_ACsperwave_vs_density_day_Np,('density', 'day','Np'))
                    update_priority.append('ACsperwave')

            if paras['control_density']:
                update_priority.append('ACtot')     # Update ACtot last
            else:
                update_priority.append('density')   # Update density last

    # --------------- Network stuff --------------#
    if paras['G']!=None:
        paras['G'].choose_short(paras['Nsp_nav'])

    # Expand or reduce capacities:
    if paras['capacity_factor']!=1.:
        for n in paras['G'].nodes():
            paras['G'].node[n]['capacity'] = int(paras['G'].node[n]['capacity']*paras['capacity_factor'])
            #print "Capacity sector", n, ":", paras['G'].node[n]['capacity']

    # ------------------- From M0 to M1 ----------------------- #
    if paras['mode_M1'] == 'standard':
        paras['STS'] = None
    else: 
        paras['N_shocks'] = 0

    paras['N_shocks'] = int(paras['N_shocks'])

    # ------------ Building of AC --------------- #

    def _func_AC(a, b):
        return [int(a*b),b-int(a*b)]  

    paras['AC']=_func_AC(paras['nA'], paras['ACtot'])               #number of air companies/operators

    def _func_AC_dict(a, b, c):
        if c[0]==c[1]:
            return {c[0]:int(a*b)}
        else:
            return {c[0]:int(a*b), c[1]:b-int(a*b)}  

    paras['AC_dict']=_func_AC_dict(paras['nA'], paras['ACtot'], paras['par'])                #number of air companies/operators


    # ------------ Building paras dictionary ---------- #


    paras.to_update = to_update

    paras.to_update['AC'] = (_func_AC,('nA', 'ACtot'))
    paras.to_update['AC_dict'] = (_func_AC_dict,('nA', 'ACtot', 'par'))

    # Add update priority here

    paras.update_priority=update_priority

    paras.analyse_dependance()

    return paras

# ============================================================================ #

"""
Functions of dependance between variables.
"""
def _func_density_vs_ACtot_na_day(ACtot, na, day):
    """
    Used to compute density when ACtot, na or day are variables.
    """
    return ACtot*na/float(day)

def _func_density_vs_ACsperwave_Np_na_day(ACsperwave, Np, ACtot, na, day):
    ACtot = _func_ACtot_vs_ACsperwave_Np(ACsperwave, Np)
    return _func_density_vs_ACtot_na_day(ACtot, na, day)

def _func_ACtot_vs_ACsperwave_Np(ACsperwave, Np):
    """
    Used to compute ACtot when ACsperwave or Np are variables.
    """
    return int(ACsperwave*Np)

def _func_ACsperwave_vs_density_day_Np(density, day, Np):
    """
    Used to compute ACsperwave when density, day or Np are variables.
    """
    return int(float(density*day/unit)/float(Np))

def _func_ACtot_vs_density_day_na(density, day, na):
    """
    Used to compute ACtot when density, day or na are variables.
    """
    return int(density*day/float(na))

def _func_Np(day, width_peak, Delta_t):
    """
    Used to compute Np based on width of waves, duration of day and 
    time between the end of a wave and the beginning of the nesx wave.
    """
    return int(ceil(day/float(width_peak+Delta_t)))

# ============================================================================ #
# ============================== Trajectories ================================ #
# ============================================================================ #

"""
All the following functions are used to manipulate the trajectories in different
formats. To represent the different formats, we use x for latitude (or first 
coordinate), y for longitude (or second coordinate), z for altitude (or third
coordinate), t for time (see different formats of times thereafter), s for the 
label of the sector in which the point lies, n for the label of the point.

The formats can essentially be:
 -- (x, y, z, t) : trajectories are made of 4-tuples with lat, lon, altitude and 
time
 -- (x, y, z, t, s) : same with sector as fifth element.
 -- (n), t : trajectories are made ONE 2-tuple. The first one is a list of labels
of nodes, the second one is the time of entrance, i.e. the time of the first 
point
 -- (n, z), t : same with altitude attached to each point.

The format of time can be either:
 -- a float representing the number of minutes elapsed since the beginning of 
the day (which is stored somewhere else). This format is denoted t.
 -- a list of tuple [yy, mm, dd, h, m , s]. This format is denoted tt.

"""

def compute_M1_trajectories(queue, starting_date):
    """
    Returns some trajectories (navpoint names) based on the given queue. 
    All altitudes are set to 0.

    Parameters
    ----------
    queue : list of Flight objects.
    starting_date : tuple or list of ints.
        Format should be [yy, mm, dd, h, m , s]

    Returns
    -------
    trajectories_nav : list
        with output format (n), tt

    Notes
    -----
    TODO: there might be a mistake here, because date_st has at least one bug. 
    datetime should be used instead.

    """
    
    trajectories_nav = []

    for f in queue:
        try:
            # Find the accepted flight plan, select the trajectory in navpoints.
            accepted_FP = f.FPs[[fpp.accepted for fpp in f.FPs].index(True)]
            trajectories_nav.append((accepted_FP.p_nav, date_st(accepted_FP.t*60., starting_date=starting_date))) 
        except ValueError:
            # If no flight plan has been accepted for this flight, skip it.
            pass

    return trajectories_nav

def convert_trajectories(G, trajectories, fmt_in='(n), t', **kwargs):
    """
    General converter of format of trajectories. The output is meant to be 
    compliant with the tactical ABM, i.e either (x, y, z, t) or (x, y, z, t, s).

    Parameters
    ----------
    G : Net object
        Needed in order to compute travel times between nodes.
    trajectories : list
        of trajectories in diverse format
    fmt_in : string
        Format of input.
    kwargs : additional parameters
        passed to other methods.

    Returns
    -------
    trajectories : list
        of converted trajectories with signature (x, y, z, t) or (x, y, z, t, s)
    Notes
    -----
    Needs expansion to support other conversion. Maybe make a class.
    Needs to specify format of output.

    """
    
    if fmt_in=='(n), t':
        return convert_trajectories_no_alt(G, trajectories, **kwargs)
    elif fmt_in=='(n, z), t':
        return convert_trajectories_alt(G, trajectories, **kwargs)
    else:
        raise Exception("format", fmt, "is not implemented")

def convert_trajectories_no_alt(G, trajectories, put_sectors=False, input_minutes=False,
    remove_flights_after_midnight=False, starting_date=[2010, 5, 6, 0, 0, 0]):
    """
    Convert trajectories with navpoint names into trajectories with coordinate and time stamps.

    trajectories signature in input:
    (n), t
    trajectories signature in output:
    (x, y, 0, t) or (x, y, 0, t, s)

    Altitudes in output are all set to 0.

    Parameters
    ----------
    G : Net object
        Used to have the coordinates of points and the travel times between nodes.
    trajectories : list
    put_sectors : boolean, optional
        If True, output format is (x, y, 0, t, s)
    input_minutes : boolean, optional
        Used to cope with the fact that the coordinates stored in the network can be in 
        degree or minutes of degree.
    remove_flights_after_midnight : boolean, True
        if True, remove from the list all flights landing ther day after starting_date
    starting_date : list of tuple 
        of format [yy, mm, dd, h, m , s]

    Returns
    -------
    trajectories_coords : list
        of trajectories with format (x, y, 0, t) or (x, y, 0, t, s)
    
    """ 

    trajectories_coords = []
    for i, (trajectory, d_t) in enumerate(trajectories):
        traj_coords = []
        for j, n in enumerate(trajectory):
            if not input_minutes:
                x = G.node[n]['coord'][0]
                y = G.node[n]['coord'][1]
            else:
                x = G.node[n]['coord'][0]/60.
                y = G.node[n]['coord'][1]/60.
            t = d_t if j==0 else date_st(delay(t) + 60.*G[n][trajectory[j-1]]['weight'])
            if remove_flights_after_midnight and list(t[:3])!=list(starting_date[:3]):
                break
            if not put_sectors:
                traj_coords.append([x, y, 0., t])
            else:
                traj_coords.append([x, y, 0., t, G.node[n]['sec']])
        if not remove_flights_after_midnight or list(t[:3])==list(starting_date[:3]):
            trajectories_coords.append(traj_coords)

    if remove_flights_after_midnight:
        print "Dropped", len(trajectories) - len(trajectories_coords), "flights because they arrive after midnight."
    return trajectories_coords

def convert_trajectories_alt(G, trajectories, put_sectors=False, input_minutes=False,
    remove_flights_after_midnight=False, starting_date=[2010, 5, 6, 0, 0, 0]):
    """
    Convert trajectories with navpoint names into trajectories with coordinate and time stamps.
    
    trajectories signature in input:
    (n, z), t
    trajectories signature in output:
    (x, y, z, t) or (x, y, z, t, s)

    Parameters
    ----------
    G : Net object
        Used to have the coordinates of points and the travel times between nodes.
    trajectories : list
    put_sectors : boolean, optional
        If True, output format is (x, y, z, t, s)
    input_minutes : boolean, optional
        Used to cope with the fact that the coordinates stored in the network can be in 
        degree or minutes of degree.
    remove_flights_after_midnight : boolean, True
        if True, remove from the list all flights landing ther day after starting_date
    starting_date : list of tuple 
        of format [yy, mm, dd, h, m , s]

    Returns
    -------
    trajectories_coords : list
        of trajectories with format (x, y, z, t) or (x, y, z, t, s)

    """ 

    trajectories_coords = []
    for i, (trajectory, d_t) in enumerate(trajectories):
        traj_coords = []
        for j, (n, z) in enumerate(trajectory):
            if not input_minutes:
                x = G.node[n]['coord'][0]
                y = G.node[n]['coord'][1]
            else:
                x = G.node[n]['coord'][0]/60.
                y = G.node[n]['coord'][1]/60.
            t = d_t if j==0 else date_st(delay(t) + 60.*G[n][trajectory[j-1][0]]['weight'])
            if remove_flights_after_midnight and list(t[:3])!=list(starting_date[:3]):
                break
            if not put_sectors:
                traj_coords.append([x, y, z, t])
            else:
                if 'sec' in G.node[n].keys():
                    sec = G.node[n]['sec']
                else:
                    sec = 0
                traj_coords.append([x, y, z, t, sec])
        if not remove_flights_after_midnight or list(t[:3])==list(starting_date[:3]):
            trajectories_coords.append(traj_coords)

    if remove_flights_after_midnight:
        print "Dropped", len(trajectories) - len(trajectories_coords), "flights because they arrive after midnight."
    return trajectories_coords

def convert_distance_trajectories(G_nav, flights):
    """
    Convert trajectories from Distance library into trajectories for strategic model. 
    Use integers for navpoints.

    Parameters
    ----------
    G_nav : NavpointNet object
    flights : list 
        of dictionnary having the 'route_m1' key. 

    Returns
    -------
    list of trajectories with format 
    """

    return [[G_nav.idx_nodes[nav] for nav, alt in flight['route_m1']] for flight in flights]

def convert_distance_trajectories_coords(G_nav, flights, put_sectors=False):
    """
    Convert trajectories from Distance library into trajectories based on coordinates. 
    Preserve the altitude and the times.
    """
    trajectories = []
    for flight in flights:
        traj = []
        for i, (nav, alt) in enumerate(flight['route_m1']):
            x, y = tuple(G_nav.node[G_nav.idx_nodes[nav]]['coord'])
            t = flight['route_m1t'][i][0] # TODO: Check this.
            if put_sectors:
                traj.append((x, y, alt, t, G_nav.node[G_nav.idx_nodes[nav]]['sec']))
            else:
                traj.append((x, y, alt, t))
        trajectories.append(traj)

    return trajectories

def write_trajectories_for_tact(trajectories, fil='../trajectories/trajectories.dat'):
    """
    Write a set of trajectories in the format for abm_tactical.
    Note: counts begin at 1 to comply with older trajectories.
    Signature trajectories:
    (x, y, z, t) or (x, y, z, t, s)
    """ 
    os.system("mkdir -p " + os.path.dirname(fil))
    with open(fil, 'w') as f:
        print >>f, str(len(trajectories)) + "\tNflights"
        for i,trajectory in enumerate(trajectories):
            print >>f, str(i+1) + "\t" + str(len(trajectory)) + '\t',
            if len(trajectory[0])==4:
                for x, y, z, t in trajectory:
                    print >>f, str(x) + "," + str(y) + "," + str(int(z)) + "," + date_abm_tactic(t) + '\t',
            else:
                for x, y, z, t, sec in trajectory:
                    print >>f, str(x) + "," + str(y) + "," + str(int(z)) + "," + date_abm_tactic(t) + ',' + str(sec) + '\t',
            print >>f, ''

    print "Trajectories saved in", fil  

def date_abm_tactic(date):
    """
    Transform a list [year, month, day, hours, minutes, seconds] in 
    YYYY-MM-DD H:mm:s:0
    """
    year, month, day, hours, minutes, seconds = tuple(date)
    month = str(month) if month>=10 else "0" + str(month)
    day = str(day) if day>=10 else "0" + str(day)

    date_abm = date_human([str(year), month, day, str(hours), str(minutes), str(seconds)]) + ':0'
    pouet = split(date_abm,'_')# replace _ by a space
    date_abm = pouet[0] + ' ' + pouet[1]
    return date_abm

# ============================================================================ #
# ============================ Complexity Utilities ========================== #
# ============================================================================ #

def restrict_to_connected_components(G):
    """
    Remove all nodes which are not in the biggest
    connected component.

    Parameters
    ----------
    G : networkx Graph of DiGraph

    Returns
    -------
    G : networkx Graph of DiGraph
        with removed nodes
    removed : list
        of labels of removed nodes.

    """

    CC = nx.connected_component_subgraphs(G)[0]
    removed = []
    for n in G.nodes()[:]:
        if not n in CC.nodes():
            G.remove_node(n)
            removed.append(n)
    return G, removed

def clean_network(G):
    """
    Remove all nodes with degree 0 from a networkx object.

    Parameters
    ----------
    G : networkx Graph of DiGraph

    Returns
    -------
    G : networkx Graph of DiGraph
        with removed nodes
    removed : list
        of labels of removed nodes

    """

    removed = []
    for n in G.nodes()[:]:
        if G.degree(n)==0:
            G.remove_node(n)
            removed.append(n)
    return G, removed

def select_interesting_navpoints(G, OD=None, N_per_sector=1, metric="centrality"):
    """
    Select N_per_sector "interesting" navpoints per sector, according to
    the metric given as input. The function selects the N_per_sector nodes 
    which have the higher metric within each sector.

    Parameters
    ----------
    G : hybrid network
    OD : list of 2-tuples, optional
        list of origin-destination nodes to use to compute the betweenness
        centrality. If None is given, the betweenness is computed on all
        possible pairs.
    N_per_sector : int, optional
        Number of interesting points to select per sector.
    metric : string, optional
        For now, only centrality is implemented.

    Raises
    ------
    Exception
        If an unknown metric is given as input.

    Returns
    -------
    n_best : dictionary
        Keys are labels of sec-node and values are lists of labels of nav-nodes.

    Notes
    -----
    Shall we compute the metric on the subnetwork of the sector?
 
    """

    if metric=="centrality":
        print "Computing betweenness centrality (between", len(OD), "pairs) ..."
        bet = bet_OD(G.G_nav, OD=OD)
    else:
        raise Exception("Metric", metric, "is not implemented.")

    # For each sector, sort the navpoints in increasing centrality and select the N_per_sector last
    n_best = {sec:[G.node[sec]['navs'][idx] for idx in np.argsort([bet[nav] for nav in G.node[sec]['navs']])[-N_per_sector:]] for sec in G.nodes()}
    
    return n_best

def select_interesting_navpoints_per_trajectory(trajs, G, OD=None, N_per_sec_per_traj=1, metric="centrality"):
    """
    Equivalent of select_interesting_navpoints for trajectories.
    Select N_per_sec_per_traj interesting napvoint per trajectory and per sector.

    Parameters
    ----------
    G : hybrid network
    OD : list of 2-tuples, optional
        list of origin-destination nodes to use to compute the betweenness
        centrality. If None is given, the betweenness is computed on all
        possible pairs.
    N_per_sector : int, optional
        Number of interesting points to select per sector.
    metric : string, optional
        For now, only centrality is implemented.

    Raises
    ------
    Exception
        If an unknown metric is given as input.

    Returns
    -------
    n_best : dictionary
        Keys are labels of sec-node and values are lists of labels of nav-nodes.

    """

    try:
        assert hasattr(G, "G_nav")
    except AssertionError:
        raise Exception("Need an hybrid network (sectors+navpoints) in input.")

    if metric=="centrality":
        print "Computing betweenness centrality (between", len(OD), "pairs) ..."
        bet = bet_OD(G.G_nav, OD=OD)
    else:
        raise Exception("Metric", metric, "is not implemented.")

    n_best = {}
    all_secs = set() # Only for information
    for traj in trajs:
        # Compute the sectors in trajectory
        secs = set([G.G_nav.node[n]['sec'] for n in traj])
        all_secs.union(secs)
        #print "secs", secs
        # For each sector, select the N_per_sec_per_traj best navpoints.
        for sec in secs:
            navs = [nav for nav in traj if G.G_nav.node[nav]['sec']==sec]
            # print "navs:", navs
            # print "bets:", [bet[nav] for nav in navs]
            # print "idx:", np.argsort([bet[nav] for nav in navs])
            # print "best:", [navs[idx] for idx in np.argsort([bet[nav] for nav in navs])[-N_per_sec_per_traj:]]
            #for n in np.argsort([bet[nav] for nav in traj if G.G_nav.node[nav]['sec']==sec])[-N_per_sec_per_traj:]
        
            n_best[sec] = n_best.get(sec, []) + [navs[idx] for idx in np.argsort([bet[nav] for nav in navs])[-N_per_sec_per_traj:]]
        #print "n_best:", n_best
        #print 
    # Remove redundant navpoints
    n_best = {sec:list(set(navs)) for sec, navs in n_best.items()}
    print "In average, I selected", np.mean([len(navs) for navs in n_best.values()]), "navpoint per sector."
    return n_best

def OD(trajectories):
    """
    Return the Origin-Destination pairs based on the trajectories.

    Format of input: (x, y, *)

    TODO: check this. How come that we use t[1] instead of t[-1]?
    """
    
    raise Warning("This function (OD) has not been tested properly.")

    return set([(t[0], t[1]) for t in trajectories])

# def insert_altitudes(trajectories, sample_trajectories):
#     """
#     To use after convert_trajectories to generate altitude based on given sample.
#     Generate constant altitudes for now. TODO: generate non-constant altitudes.
#     """
#     for traj in trajectories:
#         alt = choice(sample_trajectories) # NO!
#         for i, (x, y, z, t) in enumerate(traj):
#             traj[i] = (x, y, alt, t)

def select_heigths(th):
    """
    Sorts the altitude th increasingly, decreasingly or half/half at random.

    Parameters
    ----------
    th : list of floats.

    """
    
    coin=choice(['up','down','both'])
    if coin=='up' : th.sort()
    if coin=='down': th.sort(reverse=True)
    if coin=='both':
        a=th[:len(th)/2]
        a.sort()
        b=th[len(th)/2:]
        b.sort(reverse=True)
        th=a+b
    return th

def insert_altitudes(trajectories, sample_trajectories, min_FL=240.):
    """
    Insert altitudes in trajectories based on distribution extracted from sample_trajectories.
    The vertical trajectories can be of three kinds: increasing, decreasing, or increasing then decreasing.

    Parameters
    ----------
    trajectories : list
        of trajectories with signature (x, y, z, t) or (x, y, z, t, s)
    sample_trajectories : list
        of trajectories with signature (x, y, z, t) or (x, y, z, t, s)
    min_FL : float, optional
        cutoff for trajectories. All point below this will be discarded.

    Returns
    -------
    trajectories : list
        mofified list with injected bootstrapped altitudes.

    """

    # Detect if the trajectories have a fifth component corresponding to sector.
    sectors = len(trajectories[0][0]) == 5
    # Angles of real flights with respect to horizontal (between -pi and +pi).
    entry_exit = [(t[0], t[-1]) for t in trajectories]
    
    if not sectors:
        angles = [atan2(-(y1-y2),(x1-x2)) for (x1, y1, z1, t1), (x2, y2, z2, t2) in entry_exit]
    else:
        angles = [atan2(-(y1-y2),(x1-x2)) for (x1, y1, z1, t1, sec1), (x2, y2, z2, t2, sec2) in entry_exit]

    # Sample the heights
    if not sectors:
        h = [(int(z)/10)*10. for a in sample_trajectories for x, y, z, t in a if z>=min_FL]
    else:
        h = [(int(z)/10)*10. for a in sample_trajectories for x, y, z, t, sec in a if z>=min_FL]
    hp = [a for a in h if a%20==0] # This is for putting half flights on odd FL and the other half on even FLs.
    hd = [a for a in h if a%20!=0]
    h = [hp, hd]
    
    # Put new altitudes in trajectories
    for idx, traj in enumerate(trajectories):
        th = select_heigths([choice(h[angles[idx]<0]) for j in range(len(traj))])
        if not sectors:
            trajectories[idx] = [(x, y, th[j], t) for i, (x, y, z, t) in enumerate(traj)]
        else:
            trajectories[idx] = [(x, y, th[j], t, sec) for i, (x, y, z, t, sec) in enumerate(traj)]
        #print "Altitude picked:",th[j] 
    return trajectories
   
def iter_partial_rectification(trajectories, eff_targets, G, metric='centrality', N_per_sector=1, **kwargs_rectificate):
    """
    Used to iterate a partial_rectification without recomputing the best nodes each time.
    Working but probematic from a theoretical point of view.
    """
    trajectories_copy = deepcopy(trajectories)
    G_copy = deepcopy(G)
    # Make groups
    #n_best = select_interesting_navpoints(G, OD=OD(trajectories), N_per_sector=N_per_sector, metric=metric) # Selecting points with highest betweenness centrality within each sector
    n_best = select_interesting_navpoints_per_trajectory(trajectories_copy, G_copy, OD=OD(trajectories_copy), N_per_sec_per_traj=N_per_sector, metric=metric) # Selecting points with highest betweenness centrality within each sector
    n_best = [n for sec, points in n_best.items() for n in points]
    print 'n_best', n_best

    groups = {"C":[], "N":[]} # C for "critical", N for "normal"
    for n in G_copy.G_nav.nodes():
        if n in n_best:
            groups["C"].append(n)
        else:
            groups["N"].append(n)
    probabilities = {"C":0., "N":1.} # Fix nodes with best score (critical points).

    final_trajs_list, final_eff_list, final_G_list, final_groups_list = [], [], [], []
    for eff_target in eff_targets:
        # print "Trajectories:"
        # for traj in trajectories_copy:
        #   print traj
        final_trajs, final_eff, final_G, final_groups = rectificate_trajectories_network(trajectories_copy, eff_target, G_copy.G_nav, groups=groups, probabilities=probabilities,\
            remove_nodes=True, **kwargs_rectificate)
        for new_el, listt in [(final_trajs, final_trajs_list), (final_eff, final_eff_list), (final_G, final_G_list), (final_groups, final_groups_list)]:
            listt.append(deepcopy(new_el))
        print 

    return final_trajs_list, final_eff_list, final_G_list, final_groups_list, n_best

def partial_rectification(trajectories, eff_target, G, metric='centrality', N_per_sector=1, **kwargs_rectificate):
    """
    High level function for rectification. Fix completely N_per_sector points with 
    highest metric value per sector.
    Working but probematic from a theoretical point of view.
    """
    # Make groups
    #n_best = select_interesting_navpoints(G, OD=OD(trajectories), N_per_sector=N_per_sector, metric=metric) # Selecting points with highest betweenness centrality within each sector
    n_best = select_interesting_navpoints_per_trajectory(trajectories, G, OD=OD(trajectories), N_per_sec_per_traj=N_per_sector, metric=metric) # Selecting points with highest betweenness centrality within each sector
    
    n_best = [n for sec, points in n_best.items() for n in points]

    groups = {"C":[], "N":[]} # C for "critical", N for "normal"
    for n in G.G_nav.nodes():
        if n in n_best:
            groups["C"].append(n)
        else:
            groups["N"].append(n)
    probabilities = {"C":0., "N":1.} # Fix nodes with best score (critical points).
    
    final_trajs, final_eff, final_G, final_groups = rectificate_trajectories_network(trajectories, eff_target, G.G_nav, remove_nodes=True, 
                                                                                    groups=groups, probabilities=probabilities, **kwargs_rectificate)

    return final_trajs, final_eff, final_G, final_groups

##################################################################################
"""
Functions for checking various things
"""

def check_nav_paths(queue):
    print "Checknig references to nav paths in queue..."
    for i in range(len(queue)):
        f1 = queue[i]
        for j in range(i+1, len(queue)):
            f2 = queue[j]
            for fp1 in f1.FPs:
                for fp2 in f2.FPs:
                    try:
                        assert not fp1.p_nav is fp2.p_nav
                    except:
                        print "Nav paths of flight plans", fp1, "and", fp2, "of", f1, "and", f2, "point to the same object"
                        raise     

def check_nav_paths2(ACs):
    print "Checknig references to nav paths in ACs..."
    acs = ACs.values()

    flights = [f for ac in ACs.values() for f in ac.flights]


    for i in range(len(flights)):
        f1 = flights[i]
        for j in range(i+1, len(flights)):
            f2 = flights[j]
            for fp1 in f1.FPs:
                for fp2 in f2.FPs:
                    try:
                        assert not fp1.p_nav is fp2.p_nav
                    except:
                        print "Nav paths of flight plans", fp1, "and", fp2, "of", f1, "and", f2, "point to the same object"
                        raise     

# ============================================================================ #
# ================================== Misc. =================================== #
# ============================================================================ #

def network_whose_name_is(name):
    with open(name + '.pic') as _f:
        B = pickle.load(_f)
    return B