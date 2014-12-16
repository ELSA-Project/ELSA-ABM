# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:24:00 2013

@author: earendil

Utilies for the ABM. TODO: use general tools?
"""

import os
from mpl_toolkits.basemap import Basemap
from math import sqrt, cos, sin, pi
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

from general_tools import  delay, date_human, date_st
from tools_airports import bet_OD
version='2.9.1'

#seed(3)
_colors=['Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed','Red']

#shuffle(_colors)

def draw_network_map(G, title='Network map', trajectories=[], rep='./',airports=True, load=True, generated=False, add_to_title='', polygons=[], numbers=False, show=True):
    print "Drawing network..."
    x_min=min([G.node[n]['coord'][0]/60. for n in G.nodes()])-0.5
    x_max=max([G.node[n]['coord'][0]/60. for n in G.nodes()])+0.5
    y_min=min([G.node[n]['coord'][1]/60. for n in G.nodes()])-0.5
    y_max=max([G.node[n]['coord'][1]/60. for n in G.nodes()])+0.5
    

    #(x_min,y_min,x_max,y_max),G,airports,max_wei,zone_geo = rest
    fig=plt.figure(figsize=(9,6))#*(y_max-y_min)/(x_max-x_min)))#,dpi=600)
    gs = gridspec.GridSpec(1, 2,width_ratios=[6.,1.])
    ax = plt.subplot(gs[0])
    ax.set_aspect(1./0.8)

    
    if generated:
        def m(a,b):
            return a,b
        y,x=[G.node[n]['coord'][0] for n in G.nodes()], [G.node[n]['coord'][1] for n in G.nodes()]
    else:
        m=draw_zonemap(x_min,y_min,x_max,y_max,'i')
        x,y=split_coords(G,G.nodes(),r=0.08)
    
    for i,pol in enumerate(polygons):
        patch = PolygonPatch(pol,alpha=0.5, zorder=2, color=_colors[i%len(_colors)])
        ax.add_patch(patch) 

    if load:
        sze=[(np.average([G.node[n]['load'][i][1] for i in range(len(G.node[n]['load'])-1)],\
        weights=[(G.node[n]['load'][i+1][0] - G.node[n]['load'][i][0]) for i in range(len(G.node[n]['load'])-1)])
        /float(G.node[n]['capacity'])*800 + 5) for n in G.nodes()]
    else:
        sze=10
        
    coords={n:m(y[i],x[i]) for i,n in enumerate(G.nodes())}
    
    ax.set_title(title)
    sca=ax.scatter([coords[n][0] for n in G.nodes()],[coords[n][1] for n in G.nodes()],marker='o',zorder=6,s=sze,c='b')#,s=snf,lw=0,c=[0.,0.45,0.,1])
    if airports:
        scairports=ax.scatter([coords[n][0] for n in G.airports],[coords[n][1] for n in G.airports],marker='o',zorder=6,s=20,c='r')#,s=snf,lw=0,c=[0.,0.45,0.,1])

    if 1:
        for e in G.edges():
            plt.plot([coords[e[0]][0],coords[e[1]][0]],[coords[e[0]][1],coords[e[1]][1]],'k-',lw=0.5)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
          
    #weights={n:{v:0. for v in G.neighbors(n)} for n in G.nodes()}
    weights={n:{} for n in G.nodes()}
    for path in trajectories:
        try:
            #path=f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p
            for i in range(0,len(path)-1):
                #print path[i], path[i+1]
                #weights[path[i]][path[i+1]]+=1.
                weights[path[i]][path[i+1]] = weights[path[i]].get(path[i+1], 0.) + 1.
        except ValueError: # Why?
            pass
    
    max_w=np.max([w for vois in weights.values() for w in vois.values()])
     
    for n,vois in weights.items():
        for v,w in vois.items():
           # if G.node[n]['m1'] and G.node[v]['m1']:
                plt.plot([coords[n][0],coords[v][0]],[coords[n][1],coords[v][1]],'r-',lw=w/max_w*4.)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)

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

def split_coords(G,nodes,r=0.04):
    lines=[]
    for n in G.nodes():
        if n in nodes:
            added=False
            for l in lines:
                if sqrt((G.node[n]['coord'][0] - G.node[l[0]]['coord'][0])**2 + (G.node[n]['coord'][1] - G.node[l[0]]['coord'][1])**2)<1.: #nodes closer than 0.1 degree
                    l.append(n)
                    added=True
            if not added:
                lines.append([n])
    
    for l in lines[:]:
        if len(l)==1:
            lines.remove(l)

    pouet={}
    for l in lines:
        for n in l:
            pouet[n]=l
    x,y=[],[]
    for n in nodes:
        if not n in pouet.keys():
            x.append(G.node[n]['coord'][0]/60.)
            y.append(G.node[n]['coord'][1]/60.)
        else:
            l=pouet[n]
            theta=2.*pi*float(l.index(n))/float(len(l))
            x.append(G.node[n]['coord'][0]/60. + r*cos(theta))
            y.append(G.node[n]['coord'][1]/60. + r*sin(theta))
    return x,y
    
def draw_zonemap(x_min,y_min,x_max,y_max,res):
    m = Basemap(projection='gall',lon_0=0.,llcrnrlon=y_min,llcrnrlat=x_min,urcrnrlon=y_max,urcrnrlat=x_max,resolution=res)
    m.drawmapboundary(fill_color='white') #set a background colour
    m.fillcontinents(color='white',lake_color='white')  # #85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=0.8)
    m.drawcountries(color='#6D5F47', linewidth=0.8)
    m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb')
    m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')
    return 

def restrict_to_connected_components(G):
    """
    Remove all isolated nodes of a networkx graph.
    """
    CC=nx.connected_component_subgraphs(G)[0]
    for n in G.nodes():
        if not n in CC.nodes():
            G.remove_node(n)
    return G

class Paras(dict):
    """
    Class Paras
    =========
    Custom dictionnary used to update parameters in a controlled way.
    """
    def __init__(self, dic):
        for k,v in dic.items():
            self[k]=v
        self.to_update={}

    def update(self, name_para, new_value):
        """
        Changed in 2.9.4: self.update_priority instead of update_priority.
        """
        self[name_para]=new_value
        # Everything before level_of_priority_required should not be updated, given the para being updated.
        lvl = self.levels.get(name_para, len(self.update_priority)) #level_of_priority_required
        #print name_para, 'being updated'
        #print 'level of priority:', lvl, (lvl==len(update_priority))*'(no update)'
        for j in range(lvl, len(self.update_priority)):
            k = self.update_priority[j]
            (f, args)=self.to_update[k]
            vals=[self[a] for a in args] 
            self[k]=f(*vals)

    def analyse_dependance(self):
        """
        Detect the first level of priority hit by a dependance in each parameter.
        Those who don't need any kind of update are not in the dictionnary.
        """
        print 'Analysing dependances of the parameter with priorities', self.update_priority
        self.levels = {}
        for i, k in enumerate(self.update_priority):
            (f,args)=self.to_update[k]
            for arg in args:
                if arg not in self.levels.keys():
                    self.levels[arg] = i
 
def network_whose_name_is(name):
    with open(name + '.pic') as _f:
        B=pickle.load(_f)
    return B

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

def compute_M1_trajectories(queue):
    """
    Returns some trajectories (navpoint names) based on the given queue. 
    All altitudes are set to 0.
    """
    trajectories_nav=[]

    for f in queue:
        try:
            # Find the accepted flight plan, select the trajectory in navpoints.
            accepted_FP = f.FPs[[fpp.accepted for fpp in f.FPs].index(True)]
            trajectories_nav.append(f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p_nav) 
        except ValueError:
            pass

    return trajectories_nav

def convert_trajectories(G, trajectories, starting_date = [2010, 6, 5, 10, 0, 0]):
    """
    Convert trajectories with navpoint names into trajectories with coordinate and time stamps.
    """ 
    trajectories_coords = []
    for i,trajectory in enumerate(trajectories):
        traj_coords = []
        for j,n in enumerate(trajectory):
            x = G.node[n]['coord'][0]
            y = G.node[n]['coord'][1]
            t = 0 if j==0 else t + G[n][trajectory[j-1]]['weight']
            traj_coords.append([x, y, 0., t])
        trajectories_coords.append(traj_coords)
    return trajectories_coords

def write_trajectories_for_tact(trajectories, fil='../trajectories/trajectories.dat', starting_date = [2010, 6, 5, 10, 0, 0]):
    """
    Write a set of trajectories in the format for abm_tactical
    @G: navpoint network.
    """ 
    os.system("mkdir -p " + os.path.dirname(fil))
    with open(fil, 'w') as f:
        print >>f, str(len(trajectories)) + "\tNflights"
        for i,trajectory in enumerate(trajectories):
            print >>f, str(i) + "\t" + str(len(trajectory)) + '\t',
            for x, y, z, t in trajectory:
                print >>f, str(x) + "," + str(y) + "," + str(z) + "," + date_abm_tactic(date_st(t, starting_date = starting_date)) + '\t',
            print >>f, ''

    print "Trajectories saved in", fil  

def read_paras(paras_file = None, post_process = True):
    if paras_file==None:
        import paras as paras_mod
    else:
        paras_mod = imp.load_source("paras", paras_file)
    paras = paras_mod.paras

    if post_process:
        paras = post_process_paras(paras)

    return paras

def read_paras_iter(paras_file = 'paras.py'):
    if paras_file==None:
        import paras_iter as paras_mod
    else:
        paras_mod = imp.load_source("paras_iter", paras_file)
    paras = paras_mod.paras

    return paras

def post_process_paras(paras):
    ##################################################################################
    ################################# Post processing ################################
    ##################################################################################
    # This is useful in case of change of parameters (in particular using iter_sim) in
    # the future, to record the dependencies between variables.
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
        with open(paras['   '], 'r') as _f:
            flights = _pickle.load(_f)
        paras['flows'] = {}
        for f in flights:
            # _entry = G.G_nav.idx_navs[f['route_m1t'][0][0]]
            # _exit = G.G_nav.idx_navs[f['route_m1t'][-1][0]]
            _entry = f['route_m1t'][0][0]
            _exit = f['route_m1t'][-1][0]
            paras['flows'][(_entry, _exit)] = paras['flows'].get((_entry, _exit),[]) + [f['route_m1t'][0][1]]

        paras['departure_times'] = 'exterior'
        paras['ACtot'] = sum([len(v) for v in paras['flows'].values()])
        paras['control_density'] = False
        density=_func_density_vs_ACtot_na_day(paras['ACtot'], na, day)

        # There is no update requisites here, because the traffic shoul dnot be changed
        # when it is extracted from data.

    else:
        paras['flows'] = {}
        paras['times']=[]
        if paras['file_times'] != None:
            if paras['departure_times']=='from_data': #TODO
                with open('times_2010_5_6.pic', 'r') as f:
                    paras['times']=_pickle.load(f)
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
                Np = _func_Np(paras['day'], width_peak, Delta_t)
                to_update['Np']=(_func_Np,('day', 'width_peak', 'Delta_t'))
                update_priority.append('Np')

                if control_ACsperwave:
                    # density/ACtot based on ACsperwave
                    paras['density'] = _func_density_vs_ACsperwave_Np_na_day(paras['ACsperwave'], Np, paras['ACtot'], paras['na'], paras['day'])
                    to_update['density']=(_func_density_vs_ACsperwave_Np_na_day,('ACsperwave', 'Np', 'ACtot', 'na', 'day'))
                    update_priority.append('density')   
                else:
                    # ACperwave based on density/ACtot
                    paras['ACsperwave']=_func_ACsperwave_vs_density_day_Np(paras['density'], paras['day'], Np)
                    to_update['ACsperwave']=(_func_ACsperwave_vs_density_day_Np,('density', 'day','Np'))
                    update_priority.append('ACsperwave')

            if paras['control_density']:
                update_priority.append('ACtot')     # Update ACtot last
            else:
                update_priority.append('density')   # Update density last

    # --------------- Network stuff --------------#
    if paras['G']!=None:
        paras['G'].choose_short(paras['Nsp_nav'])

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

def select_interesting_navpoints(G, OD=None, N_per_sector=1, metric="centrality"):
    """
    Select N_per_sector interesting navpoints per sector.
    Shall we compute the metric on the subnetwork?
    """

    try:
        assert hasattr(G, "G_nav")
    except AssertionError:
        raise Exception("Need an hybrid network (sectors+navpoints) in input.")

    if metric=="centrality":
        print "Computing betweenness centrality (", len(OD), "pairs) ..."
        bet = bet_OD(G.G_nav, OD=OD)
    else:
        raise Exception("Metric", metric, "is not implemented.")

    # For each sector, sort the navpoints in increasing centrality and selected the N_per_sector last
    n_best = {sec:np.argsort([bet[nav] for nav in G.node[sec]['navs']])[-N_per_sector:] for sec in G.nodes()}
    
    return n_best

def OD(trajectories):
    """
    Return the Origin-Destination pairs based on the trajectories.
    TODO: indirected version
    """
    
    return set([(t[0], t[1]) for t in trajectories])

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

##################################################################################
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
    return int(_ceil(day/float(width_peak+Delta_t)))

##################################################################################