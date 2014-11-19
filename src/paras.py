# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:15:30 2013

@author: earendil

Parameter file for the a single simulation.
"""

# You can safely comment these two lines, it is just for me.
from import_exterior_libs import import_ext_libs as _import_ext_libs
_import_ext_libs()

import pickle as _pickle
import sys as _sys
#import numpy as _np
#from prepare_navpoint_network import prepare_navpoint_network as _prepare_navpoint_network
from math import ceil as _ceil
#import networkx as _nx

#from tools_airports import get_paras as _get_paras, extract_flows_from_data as _extract_flows_from_data
from utilities import Paras as _Paras#, network_whose_name_is as _network_whose_name_is
#from general_tools import yes as _yes


version='2.9.5' # Forked from version 2.9.5 of ABMvars.

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
################################### Parameters ###################################
##################################################################################

# Unit of time
unit = 20. # used to translate times (from instance tau) in minutes.

# ---------------- Network Setup --------------- #
# Use None if you want to generate a new network (based on paras_G file).

file_net = None
if file_net == None:
	fixnetwork=True       #if fixnetwork='False' a new graph is generated at each iteration.

# ---------------- Companies ---------------- #

Nfp = 10 			# Number of flight plans to submit for a flight (pair of departing-arriving airports)
Nsp_nav = 10 	# Number of shortest paths of navpoints per path of sector.
na = 1  			# Number of flights (pairs of departing-arriving airports) per company. Never tested for na>1.
tau = 1.*unit	# Increment of time between two departures of flights.

# -------------- Density and times of departure patterns -------------- #

day=24.*60. 			# Total duration of the simulation in minutes.

# One can specify a file to extract flows (i.e. departure times, entry exit, densities, etc.)
# Leave None for new generation of traffic.

file_traffic = None     
if file_traffic==None:
	# These variables are not fully independent and might be overwritten depending on
	# the type of control you choose.
	ACtot=100 				# Relevant for choosing the total number of ACss.
	density=20          	# Relevant for choosing an overall density of flights
	control_density = True	# If you want to set the density rather than the number of flights.
	
	departure_times='square_waves' #departing time for each flight, for each AC

	#One can also specifiy a file only for times of departures.
	file_times = None
	if file_times == None:
		if departure_times=='square_waves':
			width_peak = unit 						# Duration of a wave
			Delta_t=unit*1      	# Time between the end of a wave and the beginning of the next one.
			ACsperwave=30			# Relevant for choosing a number of ACs per wave rather than a total number
			control_ACsperwave = True	# True if you want to control ACsperwave instead of density/ACtot

noise = 0. # noise on departures in minutes.

# ----------------- Behavioral parameters -------------- #
nA=1.                        # percentage of Flights of the first population.
par=[[1.,0.,0.001], [1.,0.,1000.]]

# ------------------ From M0 to M1 ------------------- #
# Shocks. STS = Sectors To Shut
STS = None 		# Use this if you want some particular sectors to be shut. Put "None" otherwise
if STS == None:
	N_shocks=0  # Total number of sectors to shut

# --------------------System parameters -------------------- #
parallel=False						# Parallel computation or not
old_style_allocation = False		# Don't remember what that is. Computation of load?
force = False						# force overwrite of already existing simulation (based on name of file).



##################################################################################
################################# Post processing ################################
##################################################################################
# DO NOT MODIFY THIS SECTION (unless you know what you are doing).

# This is useful in case of change of parameters (in particular using iter_sim) in
# the future, to record the dependencies between variables.
update_priority=[]
to_update={}

# -------------------- Post-processing -------------------- #

par=tuple([tuple([float(_v) for _v in _p])  for _p in par]) # This is to ensure hashable type for keys.

if file_net!=None:
	with open(file_net) as f:
		G = pickle.load(f)
else:
	G = None

if file_traffic!=None:
	with open(file_traffic, 'r') as _f:
		flights = _pickle.load(_f)
	flows = {}
	for f in flights:
		# _entry = G.G_nav.idx_navs[f['route_m1t'][0][0]]
		# _exit = G.G_nav.idx_navs[f['route_m1t'][-1][0]]
		_entry = f['route_m1t'][0][0]
		_exit = f['route_m1t'][-1][0]
		flows[(_entry, _exit)] = flows.get((_entry, _exit),[]) + [f['route_m1t'][0][1]]

	departure_times = 'exterior'
	ACtot = sum([len(v) for v in flows.values()])
	control_density = False
	density=_func_density_vs_ACtot_na_day(ACtot, na, day)

else:
	flows = {}
	times=[]
	if file_times != None:
		if departure_times=='from_data':
			with open('times_2010_5_6.pic', 'r') as f:
				times=_pickle.load(f)
	else:
		if control_density:
			# ACtot is not an independent variable and is computed thanks to density
			ACtot=_func_ACtot_vs_density_day_na(density, day, na)
			to_update['ACtot']=(_func_ACtot_vs_density_day_na, ('density', 'day', 'na'))
		else:
			# Density is not an independent variables and is computed thanks to ACtot.
			density=_func_density_vs_ACtot_na_day(ACtot, na, day)
			to_update['density']=(_func_density_vs_ACtot_na_day,('ACtot','na','day'))

		assert departure_times in ['zeros','from_data','uniform','square_waves']

		if departure_times=='from_data': # TODO
			with open('times_2010_5_6.pic', 'r') as f:
				times=_pickle.load(f)

		elif departure_times=='square_waves':
			Np = _func_Np(day, width_peak, Delta_t)
			to_update['Np']=(_func_Np,('day', 'width_peak', 'Delta_t'))
			update_priority.append('Np')

			if control_ACsperwave:
				# density/ACtot based on ACsperwave
				density = _func_density_vs_ACsperwave_Np_na_day(ACsperwave, Np, ACtot, na, day)
				to_update['density']=(_func_density_vs_ACsperwave_Np_na_day,('ACsperwave', 'Np', 'ACtot', 'na', 'day'))
				update_priority.append('density')	
			else:
				# ACperwave based on density/ACtot
				ACsperwave=_func_ACsperwave_vs_density_day_Np(density, day, Np)
				to_update['ACsperwave']=(_func_ACsperwave_vs_density_day_Np,('density', 'day','Np'))
				update_priority.append('ACsperwave')

		if control_density:
			update_priority.append('ACtot') 	# Update ACtot last
		else:
			update_priority.append('density')	# Update density last

# --------------- Network stuff --------------#
if G!=None:
	G.choose_short(Nsp_nav)

# ------------ Building of AC --------------- #

def _func_AC(a, b):
    return [int(a*b),b-int(a*b)]  

AC=_func_AC(nA, ACtot)               #number of air companies/operators

def _func_AC_dict(a, b, c):
    if c[0]==c[1]:
        return {c[0]:int(a*b)}
    else:
        return {c[0]:int(a*b), c[1]:b-int(a*b)}  

AC_dict=_func_AC_dict(nA, ACtot, par)                #number of air companies/operators


# ------------ Building paras dictionary ---------- #

paras = _Paras({k:v for k,v in vars().items() if k[:1]!='_' and k!='version' and k!='Paras' and not k in [key for key in locals().keys()
       if isinstance(locals()[key], type(_sys)) and not key.startswith('__')]})

paras.to_update=to_update

paras.to_update['AC']=(_func_AC,('nA', 'ACtot'))
paras.to_update['AC_dict']=(_func_AC_dict,('nA', 'ACtot', 'par'))

# Add update priority here

paras.update_priority=update_priority

paras.analyse_dependance()
