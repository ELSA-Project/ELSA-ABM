# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:15:30 2013

@author: earendil
"""

# You can safely comment these two lines, it is just for me.
from import_exterior_libs import import_ext_libs
import_ext_libs()

import pickle as _pickle
import sys as _sys
import numpy as _np
from prepare_navpoint_network import prepare_navpoint_network as _prepare_navpoint_network
from math import ceil as _ceil
import networkx as _nx

from tools_airports import get_paras as _get_paras, extract_flows_from_data as _extract_flows_from_data
from utilities import Paras as _Paras, network_whose_name_is as _network_whose_name_is
from general_tools import yes as _yes

version='2.9.5'

#--------------------------------------------------------------#
#--------------------------------------------------------------#

update_priority=[]
to_update={}
unit = 20. # minutes

# ---------------- Network Setup --------------- #

fixnetwork=True               #if fixnetwork='True' the graph is loaded from file and is not generated at each iteration

if fixnetwork:
    if 1:  # Take existing network
        #_f=open('LF_C_NSE.pic', 'r')
        #G=network_whose_name_is('DEL29_C5_65_20_v2')
        #G=_network_whose_name_is('LF29_RC_FL350_DA20_seed15')
        #_name_network = 'LF29_RC_FL350_DA0v2'
        _name_network = '../networks/LF29_RC_FL350_DA0v3_Strong'
        G=_network_whose_name_is(_name_network)
        print 'I use the network called', _name_network
        #G=_network_whose_name_is('LF29_R_FL350_DA2_seed15')
        
        #G=_network_whose_name_is('DEL_C_65_20')
        if 0:
            def give_airports_to_network(G, airports, Nsp_nav):
                G.give_airports_to_network(airports, Nsp_nav, name = 'DEL29_C5_' + str(airports[0]) + '_' + str(airports[1]) + '_v2')
                return G

            airports = G.airports       

            print 'Computing possible pairs of airports...'
            distance=7
            airports_iter = []
            for i,n in enumerate(G.nodes()):
                for j,m in enumerate(G.nodes()):
                    if i<j:
                        if len(_nx.shortest_path(G,n,m))==distance:
                            airports_iter.append([n,m])

            #G_iter=[_network_whose_name_is(n) for n in ['DEL_C_4A', 'DEL_C_65_20' , 'DEL_C_4A2']]#, 'DEL_C_6A']]
            print 'Number of pairs with distance', distance, ':', len(airports_iter)
            to_update['G']=(give_airports_to_network,('G', 'airports', 'Nsp_nav'))
            update_priority+=['G']

            #G = give_airports_to_network(G, airports)
    else: # Prepare a new one
        #G=_prepare_network(paras_G)
        G = _prepare_navpoint_network(paras_G) #Problem TODO
else:
    G=None
    


# ---------------- Companies ---------------- #

Nfp=10 #number of flight plans to submit for a flight (pair of departing-arriving airports)
try:
    assert G.Nfp==Nfp
except:
    raise Exception('Nfp should be the same than in the network. Nfp=', Nfp, ' ; G.Nfp=', G.Nfp)

Nsp_nav = 10
Nsp_nav_iter = range(1,11)
na=1  #number of flights (pairs of departing-arriving airports) per company

tau = 1.*unit
tau_iter=_np.arange(0.0001,1.01,0.05)           # factor for shifting in time the flight plans.


# -------------- Density and times of departure patterns -------------- #

def func_density_vs_ACtot_na_day(ACtot, na, day):
    return ACtot*na/float(day)#/unit)

def func_ACtot_vs_ACsperwave_Np(ACsperwave, Np):
            return int(ACsperwave*Np)

def func_ACsperwave_vs_density_day_Np(density, day, Np):
    return int(float(density*day/unit)/float(Np))

with_flows = True
#day=24.*unit
day=30.*60.

if with_flows:
    flows = {}
    for f in G.flights_selected:
        # _entry = G.G_nav.idx_navs[f['route_m1t'][0][0]]
        # _exit = G.G_nav.idx_navs[f['route_m1t'][-1][0]]
        _entry = f['route_m1t'][0][0]
        _exit = f['route_m1t'][-1][0]
        flows[(_entry, _exit)] = flows.get((_entry, _exit),[]) + [f['route_m1t'][0][1]]

    departure_times = 'exterior'
    ACtot = sum([len(v) for v in flows.values()])
    given_density = False

    density=func_density_vs_ACtot_na_day(ACtot, na, day)
    to_update['density']=(func_density_vs_ACtot_na_day,('ACtot','na','day'))
else:
    flows = {}
    density_iter=[2.*_i for _i in range(1,11)]
    #density_iter=[5., 10.]
    
    departure_times='square_waves' #departing time for each flight, for each AC
    assert departure_times=='zeros' or departure_times=='from_data' or departure_times=='uniform' or departure_times=='square_waves'

    def func_Np(a,b,c):
        return int(_ceil(a/float(b+c)))
        
    def func_Delta_t(a):
        return a

    times=[]
    if departure_times=='uniform':
        def func_ACtot(a,b,c):
            return int(a*b/float(c))
        
        ACtot=func_ACtot(density, day, na)
        update_priority+=['ACtot']
        to_update['ACtot']=(func_ACtot, ('density', 'day', 'na'))
    elif departure_times=='from_data':
        with open('times_2010_5_6.pic', 'r') as f:
            times=_pickle.load(f)
        ACtot=100
    elif departure_times=='zeros':
        #update_priority+=['AC', 'AC_dict']
        Delta_t=1.
        ACtot=100
        ACtot_iter=[20*i for i in range(1,11)]
    elif departure_times=='square_waves':
        #Delta_t=60.*4.
        Delta_t=unit*1.
        Delta_t=float(Delta_t)
        #Delta_t_iter=range(24)
        Delta_t_iter=_np.array([0.,1., 5., 23.])
        Delta_t_iter=_np.array(Delta_t_iter*unit)

        ACsperwave=30
        ACsperwave_iter=[10*_i for _i in range(1,11)]
        
        density=20.
        #density_iter=[1.,2.,5.,10.]
        density_iter=[1., 5., 10.]
        width_peak = unit

        given_density = True

        Np = func_Np(day, width_peak, Delta_t)
        to_update['Np']=(func_Np,('day', 'width_peak', 'Delta_t'))

        if not given_density:
            #constant ACsperwave
            ACtot=func_ACtot_vs_ACsperwave_Np(ACsperwave, Np)
            to_update['ACtot']=(func_ACtot,('ACsperwave','Np'))
            update_priority+=['Np', 'ACtot','density']#,'AC', 'AC_dict']
            density=func_density_vs_ACtot_na_day(ACtot, na, day)
            to_update['density']=(func_density_vs_ACtot_na_day,('ACtot','na','day'))
        else:
            #constant density
            update_priority+=['Np','ACsperwave', 'ACtot']#,'AC', 'AC_dict']
            ACsperwave=func_ACsperwave_vs_density_day_Np(density, day, Np)
            ACtot=func_ACtot_vs_ACsperwave_Np(ACsperwave, Np)
            to_update['ACsperwave']=(func_ACsperwave_vs_density_day_Np,('density', 'day','Np'))

update_priority+=['AC', 'AC_dict']

noise = 0. # in minutes.
#noise_iter = [0, 1, 5, 10, 30, 60, 90, 120]
noise_iter = range(0,30,2)

# ----------------- Parameters of the flights -------------- #

nA=1.                        # percentage of Flights of the AC number 1

_range1=list(_np.arange(0.02,0.1,0.02))
_range2=list(_np.arange(0.9,0.98,0.02))
_range3=list(_np.arange(0.1,0.9,0.1))
#_range4=list(_np.arange(0.,1.01,0.1))
_range4=list(_np.arange(0., 1.05, 0.1))
_range5=list(_np.arange(0., 1.,0.2))
nA_iter=_range5
#nA_iter=[0.1,0.5,0.9]

#par_iter=[[[1.,0.,0.001], [1.,0.,1000.]],[[1.,0.,1.], [1.,0.,1.]], [[1.,0.,1000.], [1.,0.,1.]]]
par_iter=[[[1.,0.,10.**_e], [1.,0.,1.]] for _e in range(-3,4)]
#par_iter=[[[1.,0.,10.**_e], [1.,0.,1.]] for _e in [-3,3]]

par_iter=tuple([tuple([tuple([float(_v) for _v in _pp])  for _pp in _p])  for _p in par_iter]) # transformation in tuple, because arrays cannot be keys for dictionaries.
par=[[1.,0.,0.001], [1.,0.,1000.]]
par=tuple([tuple([float(_v) for _v in _p])  for _p in par]) 


# ------------ Building of AC --------------- #
# Shouldn't be touched.

def func_AC(a, b):
    return [int(a*b),b-int(a*b)]  

AC=func_AC(nA, ACtot)               #number of air companies/operators


def func_AC_dict(a, b, c):
    if c[0]==c[1]:
        return {c[0]:int(a*b)}
    else:
        return {c[0]:int(a*b), c[1]:b-int(a*b)}  

AC_dict=func_AC_dict(nA, ACtot, par)                #number of air companies/operators



# ------------------ From M0 to M1 ------------------- #

mode_M1 = 'sweep' # sweep or standard
if mode_M1 == 'standard':
    N_shocks=0
    N_shocks_iter=range(0,5,1)
    STS = None
else: 
    N_shocks=0
    STS = None  #Sector to Shut
    STS_iter = G.nodes()

# --------------------System parameters -------------------- #
parallel=False
n_iter = 100                #number of iterations in the main loop 
old_style_allocation = False
force = False

# ---------------------------------------------------- #

#Some possible set of parameters.

#paras_to_loop=['par', 'ACtot']
#paras_to_loop=['par', 'Delta_t']
#paras_to_loop=['nA', 'ACtot']
#paras_to_loop=['G', 'ACtot']
#paras_to_loop=['nA', 'Delta_t']
#paras_to_loop=['density']

#paras_to_loop=['par','ACsperwave', 'Delta_t']
#paras_to_loop=['par','density', 'Delta_t']
#paras_to_loop=['nA','ACsperwave', 'Delta_t']
#paras_to_loop=['nA','density', 'Delta_t']

#####
#paras_to_loop = ['airports', 'par', 'Delta_t']
#paras_to_loop = ['par', 'Delta_t']
#paras_to_loop = ['noise']
#paras_to_loop = ['par']
#paras_to_loop = ['nA']
#paras_to_loop = ['Nsp_nav']
####

#paras_to_loop = ['par', 'N_shocks']

#paras_to_loop=['par','ACsperwave']#, '']
#paras_to_loop=['par','density']#, 'Delta_t']
#paras_to_loop=['nA','ACsperwave']#, 'Delta_t']
#paras_to_loop=['nA','Delta_t']#, 'Delta_t']
#paras_to_loop=['nA','density']#, 'Delta_t']

#paras_to_loop=['par','Delta_t']
#paras_to_loop=['par', 'N_shocks']#['nA','ACtot']#['par', 'ACtot']
#paras_to_loop=['nA','N_shocks']
paras_to_loop = ['STS']

if paras_to_loop == ['nA'] and par!=tuple([tuple([float(_v) for _v in _p])  for _p in [[1.,0.,0.001], [1.,0.,1000.]]]) :
    assert _yes('The set of par does not seem consistent with the loop on nA. Proceed?')

# -------------- Stuff if G or airports in iterated ----------- #

if 'airports' in paras_to_loop and paras_to_loop[0]!='airports':
    if not _yes("You did not put 'airports' first in the sequence!. This is going to take much more time ! Continue?"):
        _sys.exit("")
if 'G' in paras_to_loop:
    G_iter=[_network_whose_name_is(_n) for _n in ['DEL_C_4A', 'DEL_C_65_20' , 'DEL_C_4A2']]#, 'DEL_C_6A']]
    for GG in G_iter:
        GG.choose_short(Nsp_nav)
G.choose_short(Nsp_nav)


# ------------ Building of paras dictionnary ---------- #

paras = _Paras({k:v for k,v in vars().items() if k[:1]!='_' and k!='version' and k!='Paras' and not k in [key for key in locals().keys()
       if isinstance(locals()[key], type(_sys)) and not key.startswith('__')]})

paras.to_update=to_update

paras.to_update['AC']=(func_AC,('nA', 'ACtot'))
paras.to_update['AC_dict']=(func_AC_dict,('nA', 'ACtot', 'par'))

paras.update_priority=update_priority

paras.analyse_dependance()

