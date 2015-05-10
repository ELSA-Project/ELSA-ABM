#!/usr/bin/env python

import os

files = [#'test_iter_simO.py',
		'test_efficiency.py',
		#'test_abm_interface.py',
		'test_prepare_navpoint_network.py',
		'test_simAirSpaceO.py',
		'test_simulationO.py',
		'test_utilities.py']

for fil in files:
	os.system('./' + fil)
	print
	print
	print