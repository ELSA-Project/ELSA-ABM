#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest

from abm_strategic.simAirSpaceO import *

def simple_network():
	G = Net()
	G.build_nodes(5, prelist=[(0.,0.), (0., 1.), (1., 0.), (1., 1.), (0.5, 0.5)])
	G.add_edges_from([(4, 0), (4, 1), (4, 2), (4, 3)], weight = 40.)
	for i in range(5):
		G.node[i]['capacity'] = 5
	return G
################### Network Manager ########################

class NewStyleNMTest(unittest.TestCase):
	def setUp(self):
		self.NM = Network_Manager()
		self.G = simple_network()
		self.ACs = []
		self.ACs.append(AirCompany(0, 2, 1, [(0, 3)], (0.001, 1000.)))
		self.ACs.append(AirCompany(1, 2, 1, [(1, 2)], (0.001, 1000.)))
		for ac in self.ACs:
			ac.fill_FPs([0.], 20., self.G)


class NewStyleNMTestSimple(NewStyleNMTest):
	def test_allocate_flight(self):
		self.NM.allocate_flight(self.G, self, self.ACs[0].flights[0])
		self.assertTrue()

if __name__ == '__main__':
	unittest.main()
