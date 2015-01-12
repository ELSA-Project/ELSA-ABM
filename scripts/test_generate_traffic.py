#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic')
from abm_strategic.utilities import network_whose_name_is
from abm_strategic.simulationO import generate_traffic

#TODO; polish this.

#G = network_whose_name_is('../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2')
G = network_whose_name_is('../networks/Real_LF_v5.8_Strong_EXTLFMM_LFMM_2010-5-6+0_d2_cut240.0_directed')

save_file = '../trajectories/trajectories.dat'
file_traffic = '../networks/Real_LF_v5.8_Strong_EXTLFMM_LFMM_2010-5-6+0_d2_cut240.0_directed_flights_selected.pic'
generate_traffic(G, save_file=save_file, file_traffic=file_traffic, ACtot=1000)