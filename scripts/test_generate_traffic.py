#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
from abm.strategic.utilities import network_whose_name_is
from abm_strategic.simulationO import generate_traffic

#TODO; polish this.

G = network_whose_name_is('../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2')

generate_traffic(G, save_file = '../trajectories/trajectories.dat', ACtot = 10)