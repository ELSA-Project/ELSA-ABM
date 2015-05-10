#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
import os

from abm_strategic import *
from abm_strategic.iter_simO import *


if __name__ == '__main__':
	# Manual tests
	# os.system('../abm_strategic/iter_simO.py paras_iter_test.py')

	paras = read_paras_iter(paras_file='paras_iter_test.py')
	results = iter_sim(paras, save=1, rep='results_tests/')
	os.system('rm -r results_tests/')
	
	# Put failfast=True for stopping the test as soon as one test fails.
	unittest.main(failfast=True)