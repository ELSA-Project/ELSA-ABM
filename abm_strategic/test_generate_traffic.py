#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utilities import network_whose_name_is
from simulationO import generate_traffic

G = network_whose_name_is('../networks/D_N44_nairports22_cap_constant_C5_w_coords_Nfp2')

generate_traffic(G, save_file = '../trajectories/trajectories.dat', ACtot = 10)