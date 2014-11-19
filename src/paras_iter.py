# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:15:30 2013

@author: earendil

Parameter file for iterated simulations.
"""

# You can safely comment these two lines, it is just for me.
from import_exterior_libs import import_ext_libs as _import_ext_libs
_import_ext_libs()

import pickle as _pickle
import sys as _sys
import numpy as _np
#from prepare_navpoint_network import prepare_navpoint_network as _prepare_navpoint_network
from math import ceil as _ceil
#import networkx as _nx

#from tools_airports import get_paras as _get_paras, extract_flows_from_data as _extract_flows_from_data
from utilities import read_paras#, network_whose_name_is as _network_whose_name_is
from general_tools import yes as _yes

version='2.9.5' # Forked from version 2.9.5 of ABMvars.

##################################################################################
################################### Parameters ###################################
##################################################################################

paras = read_paras() # Import main parameters

if paras['file_net'] == None:
	fixnetwork=True       				#if fixnetwork='False' a new graph is generated at each iteration.

# ---------------- Companies ---------------- #

Nsp_nav_iter = range(1,11)
tau_iter=_np.arange(0.0001,1.01,0.05)           # factor for shifting in time the flight plans.

# -------------- Density and times of departure patterns -------------- #
    
if paras['file_traffic']==None:
	density_iter=[2.*_i for _i in range(1,11)]
	ACtot_iter=[20*i for i in range(1,11)]
	if paras['file_times'] == None:
		if paras['departure_times']=='square_waves':
			Delta_t_iter=_np.array([0.,1., 5., 23.])
			Delta_t_iter=_np.array(Delta_t_iter*paras['unit'])
			ACsperwave_iter=[10*_i for _i in range(1,11)]			# Relevant for choosing a number of ACs per wave rather than a total number
			
noise_iter = range(0,30,2)

# ----------------- Behavioral parameters -------------- #
_range1=list(_np.arange(0.02,0.1,0.02))
_range2=list(_np.arange(0.9,0.98,0.02))
_range3=list(_np.arange(0.1,0.9,0.1))
_range4=list(_np.arange(0., 1.05, 0.1))
_range5=list(_np.arange(0., 1.,0.2))

nA_iter=_range5

par_iter=[[[1.,0.,10.**_e], [1.,0.,1.]] for _e in range(-3,4)]

# ------------------ From M0 to M1 ------------------- #
mode_M1 = 'standard' # sweep or standard
if mode_M1 == 'standard':
	N_shocks_iter=range(0,5,1)
else: 
	STS_iter = paras['G'].nodes()

# --------------------System parameters -------------------- #
n_iter = 100 # Number of iterations for each set of parameters.

# --------------------- Paras to loop --------------- #
# Set the parameters to sweep by indicating their name. You can 
# put an empty list if you just want to have several iterations 
# of a single set of parameters.

#paras_to_loop = ['STS']

# --- Some possible sets of parameters.
#paras_to_loop=['par', 'ACtot']
#paras_to_loop=['par', 'Delta_t']
#paras_to_loop=['nA', 'ACtot']
#paras_to_loop=['G', 'ACtot']
#paras_to_loop=['nA', 'Delta_t']
paras_to_loop=['density']

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

if paras_to_loop == ['nA'] and par!=tuple([tuple([float(_v) for _v in _p])  for _p in [[1.,0.,0.001], [1.,0.,1000.]]]) :
	assert _yes('The set of par does not seem consistent with the loop on nA. Proceed?')


# -------------- Stuff if G or airports is iterated ----------- #
# You can also loop on "airports" or networks.

if 'airports' in paras_to_loop and paras_to_loop[0]!='airports':
	if not _yes("You did not put 'airports' first in the sequence!. This is going to take much more time ! Continue?"):
		_sys.exit("")

if 'G' in paras_to_loop:
	G_iter=[_network_whose_name_is(_n) for _n in ['DEL_C_4A', 'DEL_C_65_20' , 'DEL_C_4A2']]#, 'DEL_C_6A']]
	for GG in G_iter:
		GG.choose_short(paras['Nsp_nav'])

##################################################################################
################################# Post processing ################################
##################################################################################
# DO NOT MODIFY THIS SECTION.

# -------------------- Post-processing -------------------- #
# Add new parameters to the dictionary.

paras['par_iter']=tuple([tuple([tuple([float(_v) for _v in _pp])  for _pp in _p])  for _p in paras['par_iter']]) # transformation in tuple, because arrays cannot be keys for dictionaries.
paras['fixnetwork'] = paras['par_iter']

for k,v in vars().items():
    if k[-4:]=='iter' and k[:-5] in paras_to_loop:
        paras[k] = v

paras['paras_to_loop'] = paras_to_loop
