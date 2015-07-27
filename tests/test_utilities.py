#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest

from abm_strategic.utilities import *

class DrawTest(unittest.TestCase):
	def test_draw1(self):
		trajectories = []
		trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (1., 1., 0., [2010, 1, 1, 0, 10., 0])])
		trajectories.append([(0., 1., 0., [2010, 1, 1, 0, 0, 0]), (1., 1.000001, 0., [2010, 1, 1, 0, 15., 0])])
	
		draw_traffic_network(trajectories, fmt_in='(x, y, z, t)', show=False)

	def test_draw2(self):
		trajectories = []
		trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (1., 1., 0., [2010, 1, 1, 0, 10., 0])])
		trajectories.append([(0., 1., 0., [2010, 1, 1, 0, 0, 0]), (1., 1.000001, 0., [2010, 1, 1, 0, 15., 0])])
		trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (0., 1., 0., [2010, 1, 1, 0, 15., 0]), (1., 1., 0., [2010, 1, 1, 0, 20., 0])])
		
		draw_traffic_network(trajectories, fmt_in='(x, y, z, t)', show=False)

class readTrajectoriesTest(unittest.TestCase):
	def test_read1(self):
		fil = 'example/M1_example.dat'

		with open(fil, 'r') as f:
			lines = f.readlines()

		n_flights = int(lines[0].split('\t')[0])

		trajectories = read_trajectories_for_tact(fil, fmt_out='(x, y, z, t)')

		self.assertTrue(len(trajectories)==n_flights)
		self.assertTrue(trajectories[0][0][0]==46.8075)
		self.assertTrue(trajectories[0][0][1]==7.88083333333)
		self.assertTrue(trajectories[0][0][2]==380)
		self.assertTrue(trajectories[0][0][3]==(2010, 5, 6, 4, 39, 23))
		#self.assertTrue(trajectories[0][0][4]==-1)

	def test_read2(self):
		fil = 'example/M1_example.dat'

		with open(fil, 'r') as f:
			lines = f.readlines()

		n_flights = int(lines[0].split('\t')[0])

		trajectories = read_trajectories_for_tact(fil, fmt_out='(x, y, z, t, s)')

		self.assertTrue(len(trajectories)==n_flights)
		self.assertTrue(trajectories[0][0][0]==46.8075)
		self.assertTrue(trajectories[0][0][1]==7.88083333333)
		self.assertTrue(trajectories[0][0][2]==380)
		self.assertTrue(trajectories[0][0][3]==(2010, 5, 6, 4, 39, 23))
		self.assertTrue(trajectories[0][0][4]==-1)

if __name__ == '__main__':
	# Put failfast=True for stopping the test as soon as one test fails.
	unittest.main(failfast=True)
		 