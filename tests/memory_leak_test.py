#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys, os
sys.path.insert(1, '..')

from paths import result_dir
from interface.abm_interface import do_ABM_tactical

for i in range(20):
	do_ABM_tactical(result_dir + '/trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat', '/tmp/out.dat')