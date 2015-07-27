#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest

from libs.tools_airports import *

class trafficnetworkTest(unittest.TestCase):
	#def setUp(self):
	#	self.Converter = TrajConverter()

	def print_trajs(self, trajs):
		print "Trajectories:"
		for traj in trajs:
			print traj
		print

	def print_G(self, G):
		print "Network:"
		for n in G.nodes():
			print 'Node', n, ':', G.node[n]['coord']
		print

		for n1, n2 in G.edges():
			print 'Edge', n1, n2, ':', G[n1][n2]['weight']
		print


	def test1(self):
		trajectories = []
		trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (1., 1., 0., [2010, 1, 1, 0, 10., 0])])
		trajectories.append([(0., 1., 0., [2010, 1, 1, 0, 0, 0]), (1., 1., 0., [2010, 1, 1, 0, 15., 0])])
		#trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (0., 1., 0., [2010, 1, 1, 0, 15., 0]), (1., 1., 0., [2010, 1, 1, 0, 20., 0])])
		#new_trajs = self.Converter.convert(trajectories, fmt_in='(x, y, z, t)', fmt_out='(n), t')
	
		G = build_traffic_network(trajectories, fmt_in='(x, y, z, t)')

		#self.print_trajs(trajectories)

		#self.print_G(G)

		self.assertTrue(G[0][1]['weight']==1)
		self.assertTrue(G[0][2]['weight']==1)
		self.assertTrue(len(G.nodes())==3)

	def test2(self):
		trajectories = []
		trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (1., 1., 0., [2010, 1, 1, 0, 10., 0])])
		trajectories.append([(0., 1., 0., [2010, 1, 1, 0, 0, 0]), (1., 1., 0., [2010, 1, 1, 0, 15., 0])])
		trajectories.append([(0., 0., 0., [2010, 1, 1, 0, 0, 0]), (0., 1., 0., [2010, 1, 1, 0, 15., 0]), (1., 1., 0., [2010, 1, 1, 0, 20., 0])])
		#new_trajs = self.Converter.convert(trajectories, fmt_in='(x, y, z, t)', fmt_out='(n), t')
	
		G = build_traffic_network(trajectories, fmt_in='(x, y, z, t)')

		self.print_trajs(trajectories)

		self.print_G(G)

		self.assertTrue(G[0][1]['weight']==1)
		self.assertTrue(G[0][2]['weight']==1)
		self.assertTrue(G[1][2]['weight']==2)
		self.assertTrue(len(G.nodes())==3)


if __name__ == '__main__':
	# Put failfast=True for stopping the test as soon as one test fails.
	unittest.main(failfast=True)
