#!/usr/bin/env python

import sys
import os
sys.path.insert(1, '../..')
#sys.path.insert(1, '../../abm_strategic')
from interface.abm_interface import do_ABM_tactical
from abm_strategic import main_dir

from os.path import join as jn

if __name__ == '__main__':

	input_file = jn(main_dir, 'tests', 'example', 'M1_example.dat')
	output_file = jn(main_dir, 'tests', 'example', 'M3_example.dat')
	config_file = jn(main_dir, 'tests', 'example', 'config', 'config.cfg')
	shock_file = jn(main_dir, 'tests', 'example', 'config', 'shock_tmp.dat')
	bound_file = jn(main_dir, 'tests', 'example', 'config', 'bound_latlon.dat')
	tmp_navpoints_file = jn(main_dir, 'tests', 'example', 'config', 'temp_nvp.dat')
	capacity_file = jn(main_dir, 'tests', 'example', 'config', 'sector_capacities.dat')

	os.system('cp ' + config_file + ' ' + config_file + '.bk')

	try:
		do_ABM_tactical(input_file, output_file, config_file, verbose=1, 
									shock_tmp=shock_file,
									bound_latlon=bound_file,
									temp_nvp=tmp_navpoints_file,
									capacity_file=capacity_file)
	finally:
		os.system('mv ' + config_file + '.bk' + ' ' + config_file )
