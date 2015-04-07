#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
from networkx import Graph
import types
import numpy as np
from shapely.geometry import Polygon

from abm_strategic.prepare_navpoint_network import *

def assertCoordinatesAreEqual(l1, l2, thr=10**(-6.)):
	b1 = len(l1)==len(l2)
	if not b1:
		return False
	else:
		s = sum(((l1[i][0]-l2[i][0])**2 + (l1[i][1]-l2[i][1])**2) for i in range(len(l2)))
		b2 = s<thr
		return b2

class TestLowFunctions(unittest.TestCase):
	def test_area(self):
		l = [(0., 0.), (0., 2.), (2., 2.), (2., 0.), (0., 0.)]
		self.assertEqual(area(l), 4.)

	def test_segments(self):
		l = [(0., 0.), (0., 2.), (2., 2.)]
		self.assertEqual(segments(l), [((0., 0.), (0., 2.)), ((0., 2.), (2., 2.)), ((2., 2.), (0., 0.))])


class SimpleSectorNetworkCase(unittest.TestCase):

	def setUp(self):
		self.G = Graph()
		self.G.add_node(0, coord=(0., 0.))
		self.G.add_node(1, coord=(1., 0.))
		self.G.add_node(2, coord=(-0.5, np.sqrt(3.)/2.))
		self.G.add_node(3, coord=(0.5, np.sqrt(3.)/2.))
		self.G.add_node(4, coord=(1.5, np.sqrt(3.)/2.))
		self.G.add_node(5, coord=(0., np.sqrt(3.)))
		self.G.add_node(6, coord=(1., np.sqrt(3.)))

class HybridNetworkCase(unittest.TestCase):
	def setUp(self):
		#super(HybridNetworkCase, self).__init__()
		self.G = Graph()
		self.G.add_node(0, coord=(0., 0.))
		self.G.add_node(1, coord=(1., 0.))
		self.G.add_node(2, coord=(-0.5, np.sqrt(3.)/2.))
		self.G.add_node(3, coord=(0.5, np.sqrt(3.)/2.))
		self.G.add_node(4, coord=(1.5, np.sqrt(3.)/2.))
		self.G.add_node(5, coord=(0., np.sqrt(3.)))
		self.G.add_node(6, coord=(1., np.sqrt(3.)))

		self.G.G_nav = Graph()
		l = np.sqrt(3.)
		#pouet = []

		# Put six points inside the central sector
		center = np.array([0.5, l/2.])
		for i in range(6):
			angle = float(i)/6.*2.*np.pi
			point = center + 0.25*np.array([np.cos(angle), np.sin(angle)])
			#pouet.append(list(point))
			self.G.G_nav.add_node(i, coord=point)

		# Put two points in the 6 outer sectors
		for i in range(12):
			angle = 1./12.*2.*np.pi + float(i)/12.*2.*np.pi
			point = center + 0.75*np.array([np.cos(angle), np.sin(angle)])
			self.G.G_nav.add_node(6 + i, coord=point)

		self.G.G_nav.add_node(18, coord=center)

		self.G.airports = []
		self.G.G_nav.airports = []

		def get_airports(G):
			return G.airports

		self.G.get_airports = types.MethodType(get_airports, self.G)
		self.G.G_nav.get_airports = types.MethodType(get_airports, self.G.G_nav)

		self.G.G_nav.navpoints_borders = []


	def give_nodes_to_secs(self):
		self.G.node[3]['navs'] = list(range(6))
		self.G.node[0]['navs'] = [13, 14]
		self.G.node[1]['navs'] = [15, 16]
		self.G.node[2]['navs'] = [11, 12]
		self.G.node[4]['navs'] = [6, 17]
		self.G.node[5]['navs'] = [9, 10]
		self.G.node[6]['navs'] = [7, 8]
		
		for i in range(6):
			self.G.G_nav.node[i]['sec'] = 3
		self.G.G_nav.node[18]['sec'] = 3
		self.G.G_nav.node[6]['sec'] = self.G.G_nav.node[17]['sec'] = 4
		self.G.G_nav.node[7]['sec'] = self.G.G_nav.node[8]['sec'] = 6
		self.G.G_nav.node[9]['sec'] = self.G.G_nav.node[10]['sec'] = 5
		self.G.G_nav.node[11]['sec'] = self.G.G_nav.node[12]['sec'] = 2
		self.G.G_nav.node[13]['sec'] = self.G.G_nav.node[14]['sec'] = 0
		self.G.G_nav.node[15]['sec'] = self.G.G_nav.node[16]['sec'] = 1

	def give_full_connections(self):
		for i in range(7):
			if i!=3:
				self.G.add_edge(3, i)
				self.G.add_edge(i, (i+1)%6)

		for i in range(6):
			self.G.G_nav.add_edge(18, i)
			self.G.G_nav.add_edge(i, (i+1)%6)
			self.G.G_nav.add_edge(i, 2*i+6)
			coin = (2*i+5) if (2*i+5)!=5 else 17
			self.G.G_nav.add_edge(i, coin)

		for i in range(6, 18):
			coin = i+1 if i+1 != 18 else 6
		 	self.G.G_nav.add_edge(i, coin)

		#print "TEST: Creating the following edges:"
		#print self.G.G_nav.edges()



class TestLowNetworkFunctions(SimpleSectorNetworkCase):

	def test_compute_voronoi(self):
		G, vor = compute_voronoi(self.G, a=3.)
		coin = [list(pol.exterior.coords) for pol in G.polygons.values() if type(pol)==type(Polygon())]
		self.assertTrue(len(coin)==1)
		l = np.sqrt(3.)
		pouet = []
		center = np.array([0.5, l/2.])
		for i in range(6):
			angle = - np.pi/2.- float(i)/6.*2.*np.pi
			point = center + (1./l)*np.array([np.cos(angle), np.sin(angle)])
			pouet.append(list(point))
		pouet.append(pouet[0])
		self.assertTrue(len(coin[0])==7)
		self.assertTrue(assertCoordinatesAreEqual(coin[0], pouet))

	def test_reduce_airports_to_existing_nodes(self):
		airports = [0, 1, 10000]
		pairs = [(0, 1), (0, 100000), (10000, 1)]
		pairs, airports = reduce_airports_to_existing_nodes(self.G, pairs, airports)
		self.assertEqual(pairs, [(0, 1)])
		self.assertEqual(airports, [0, 1])

	def test_recompute_neighbors(self):
		G, vor = compute_voronoi(self.G, a=3.)
		#print [list(pol) for pol in G.polygons.values() if type(pol)!=type(Polygon())]
		G.add_edge(0, 6)
		#G.add_edge(0, 3)
		recompute_neighbors(G)
		self.assertFalse(G.has_edge(0,6))
		#self.assertTrue(G.has_edge(0,3))

	def test_compute_navpoints_borders(self):
		borders_coordinates = [((0., 0.), (0., 1.))]
		shape = Polygon([(-1., -1.), (-1., 2.), (2., 2.), (2., -1.)])
		navpoints = compute_navpoints_borders(borders_coordinates, shape, lin_dens=1)   
		self.assertEqual(len(navpoints), 1)
		self.assertEqual(list(navpoints[0]), [10**(-5.), 0.5 + 10**(-5.)])

	def test_navpoints_at_borders(self):
		G, vor = compute_voronoi(self.G, a=3.)
		G.polygons = {i:pol for i, pol in G.polygons.items() if type(pol)==type(Polygon())}
		G.global_shape=cascaded_union(G.polygons.values())
		l = np.sqrt(3.)
		navpoints = navpoints_at_borders(G, lin_dens=1./l)

		witness = []
		center = np.array([0.5, l/2.])
		for i in range(6):
			angle = - np.pi/2. - np.pi/6. - float(i)/6.*2.*np.pi
			point = center + 0.5*np.array([np.cos(angle), np.sin(angle)])# + 10**(-5.)
			witness.append(list(point))

		self.assertEqual(len(navpoints), len(witness))
		self.assertTrue(assertCoordinatesAreEqual(navpoints, witness, thr=len(navpoints)*10**(-5.)))

	def test_extract_weights_from_traffic(self):
		self.G.idx_nodes = {('node' + str(i)):i for i in self.G.nodes()}
		self.G.add_edges_from([(0, 3), (3, 6), (3, 4)])
		f1 = {'route_m1t':[('node0', (2010, 5, 6, 0, 0, 0)), ('node3', (2010, 5, 6, 0, 10, 0)), ('node6', (2010, 5, 6, 0, 25, 0))]}
		f2 = {'route_m1t':[('node0', (2010, 10, 6, 0, 0, 0)), ('node3', (2010, 10, 6, 0, 15, 0)), ('node4', (2010, 10, 6, 0, 35, 0))]}
		flights = [f1, f2]
		
		weights = extract_weights_from_traffic(self.G, flights)

		self.assertEqual(weights[(0, 3)], 12.5)
		self.assertEqual(weights[(3, 6)], 15.)
		self.assertEqual(weights[(3, 4)], 20.)

	def test_prepare_sectors_network(self):
		# TODO when the Net object has been tested.
		pass

class TestHighNetworkFunctions(HybridNetworkCase):
	def test_attach_termination_nodes(self):
		self.give_full_connections()
		self.give_nodes_to_secs()
		self.G.G_nav.remove_edges_from([(6, 17), (6, 7)])

		# Shift slightly the central navpoint so as to be closer to 
		# node number 0.
		self.G.G_nav.node[18]['coord'] += np.array([0.0001, 0.])
		self.G.G_nav.remove_edges_from([(18, 0), (18, 1), (18, 2), (18, 3), (18, 4)])

		self.assertTrue(self.G.G_nav.degree(18)==1)
		self.assertTrue(self.G.G_nav.degree(6)==1)

		self.G = attach_termination_nodes(self.G)

		self.assertTrue(self.G.G_nav.degree(18)==2)
		self.assertTrue(self.G.G_nav.degree(6)==2)

		self.assertTrue(17 in self.G.G_nav.neighbors(6))
		self.assertTrue(0 in self.G.G_nav.neighbors(18))

	def test_attach_two_sectors(self):
		self.give_full_connections()
		self.give_nodes_to_secs()
		
		# Detach nav-link between sectors 3 and 4
		self.G.G_nav.remove_edges_from([(0, 6), (0, 17)])

		# Shift slightly node 6 so as to be closer to node 0
		self.G.G_nav.node[6]['coord'] += np.array([0., 0.])

		print self.G.G_nav.neighbors(6)
		print self.G.G_nav.neighbors(17)

		self.G = attach_two_sectors(3, 4, self.G)

		print self.G.G_nav.neighbors(6)
		print self.G.G_nav.neighbors(17)

		self.assertTrue(self.G.G_nav.has_edge(0, 6))

		#TO FINISH

if __name__ == '__main__':
	#suite = unittest.TestLoader().loadTestsFromTestCase(TestLawNetworkFunctions)
	unittest.main()
