#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
import networkx as nx
import numpy as np
from random import seed

from abm_strategic.simAirSpaceO import *

def simple_network():
	G = Net()
	G.build_nodes(5, prelist=[(0.,0.), (0., 1.), (1., 0.), (1., 1.), (0.5, 0.5)])
	G.add_edges_from([(4, 0), (4, 1), (4, 2), (4, 3)], weight = 40.)
	for i in range(5):
		G.node[i]['capacity'] = 5
	return G

################### Network Manager ########################

# class NewStyleNMTest(unittest.TestCase):
# 	def setUp(self):
# 		self.NM = Network_Manager()
# 		self.G = simple_network()
# 		self.ACs = []
# 		self.ACs.append(AirCompany(0, 2, 1, [(0, 3)], (0.001, 1000.)))
# 		self.ACs.append(AirCompany(1, 2, 1, [(1, 2)], (0.001, 1000.)))
# 		for ac in self.ACs:
# 			ac.fill_FPs([0.], 20., self.G)


# class NewStyleNMTestSimple(NewStyleNMTest):
# 	def test_allocate_flight(self):
# 		self.NM.allocate_flight(self.G, self, self.ACs[0].flights[0])
# 		self.assertTrue()

class BuildingNetTest(unittest.TestCase):
	def setUp(self):
		self.G = nx.Graph()
		self.G.add_node(0, coord=(0., 0.))
		self.G.add_node(1, coord=(1., 0.))
		self.G.add_node(2, coord=(-0.5, np.sqrt(3.)/2.))
		self.G.add_node(3, coord=(0.5, np.sqrt(3.)/2.))
		self.G.add_node(4, coord=(1.5, np.sqrt(3.)/2.))
		self.G.add_node(5, coord=(0., np.sqrt(3.)))
		self.G.add_node(6, coord=(1., np.sqrt(3.)))

	def put_edges(self):
		self.G.add_edge(0, 1)
		self.G.add_edge(1, 4)
		self.G.add_edge(4, 6)
		self.G.add_edge(6, 5)
		self.G.add_edge(5, 2)
		self.G.add_edge(2, 0)

		for i in [0, 1, 2, 4, 5, 6]:
			self.G.add_edge(3, i)	
				
	def put_weights(self):
		for n1, n2 in self.G.edges():
			self.G[n1][n2]['weight'] = 2.

	def test_import_from(self):
		self.put_edges()
		
		N = Net()
		N.import_from(self.G)

		self.assertTrue(set(self.G.nodes())==set(N.nodes()))
		self.assertTrue(set(self.G.edges())==set(N.edges()))
		self.assertFalse(N.weighted)

		self.put_weights()

		M = Net()
		M.import_from(self.G)

		self.assertTrue(M.weighted)

		self.G.add_node('Roger')

		P = Net()
		P.import_from(self.G, numberize=False)
		self.assertTrue('Roger' in P.nodes())

		Q = Net()
		Q.import_from(self.G, numberize=True)
		self.assertFalse('Roger' in Q.nodes())
		self.assertTrue(set(range(len(Q))) == set(Q.nodes()))

	def test_build_net(self):
		N = Net()
		N.import_from(self.G)
		M = Net()
		M.import_from(self.G)

		self.put_edges()

		N.build_net(Gtype='D')

		self.assertTrue(set(self.G.edges())==set(N.edges()))

		M.build_net(Gtype='E', mean_degree=2)

		# Which test?

	def test_generate_weights(self):
		self.put_edges()
		N = Net()
		N.import_from(self.G)
		N.generate_weights(typ='coords', par=3.5)

		self.assertTrue((np.mean([N[e1][e2]['weight'] for e1, e2 in N.edges()])-3.5)<10**(-5.))

	def test_generate_capacities(self):


if __name__ == '__main__':
	unittest.main(failfast=True)
