#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic')

#from paths import result_dir
from abm_strategic.utilities import network_whose_name_is
from abm_strategic.simulationO import generate_traffic
from libs.general_tools import draw_network_and_patches

import abm_strategic
result_dir = abm_strategic.result_dir

#TODO; polish this.

#G = network_whose_name_is('../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2')
#G = network_whose_name_is('../networks/Real_LF_v5.8_Strong_EXTLFMM_LFMM_2010-5-6+0_d2_cut240.0_directed')
G = network_whose_name_is(result_dir + '/networks/Real_LI_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed')

name_G = 'Real_LI_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed'

save_file = result_dir + '/trajectories/M1/trajectories.dat'
#save_file = result_dir + '/trajectories/M1/test_rectificate.dat'
#file_traffic = '../networks/Real_LF_v5.8_Strong_EXTLFMM_LFMM_2010-5-6+0_d2_cut240.0_directed_flights_selected.pic'
file_traffic = result_dir + "/networks/" + name_G + '_flights_selected.pic'
#file_traffic = '../networks/Real_LF_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed_flights_selected.pic'
#save_file_capacities = '../networks/Real_LF_v5.8_Strong_EXTLFMM_LFMM_2010-5-6+0_d2_cut240.0_directed_flights_selected_capacities.dat'
save_file_capacities = result_dir + '/networks/Real_LI_v5.8_Strong_EXTLIRR_LIRR_2010-5-6+0_d2_cut240.0_directed_flights_selected_capacities.dat'
trajectories = generate_traffic(G, 	save_file=save_file, 
									file_traffic=file_traffic, 
									coordinates=True,
									put_sectors=True, 
									#save_file_capacities=save_file_capacities,
									capacity_factor=0.05,
									remove_flights_after_midnight=True,
									#record_stats_file='../trajectories/M1/trajectories_stats.dat'
									#rectificate={'eff_target':0.99, 'inplace':False, 'hard_fixed':False, 'remove_nodes':True, 'resample_trajectories':True},
									storymode=False,
									ACtot=4000
									)
#print trajectories
#draw_network_and_patches(None, G.G_nav, G.polygons, show=True, flip_axes=True,\
 #trajectories=trajectories, save=False)