#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
from networkx import Graph
import types
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from random import seed

from abm_strategic.prepare_navpoint_network import *

"""
TODO: fucntions with traffic, big function prepare_hybrid_network and 
prepare_sectors_network
"""

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

	def show_network(self, show=True):
		plt.scatter(*zip(*[self.G.node[n]['coord'] for n in self.G.nodes()]), marker='s', color='r', s=50)

		if show:
			plt.show()

	def show_polygons(self, show=True):
		for pol in self.G.polygons.values():
			plt.fill(*zip(*list(pol.exterior.coords)), alpha=0.4)

		if show:
			plt.show()

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
			angle = 1./24.*(2.*np.pi) + float(i)/12.*(2.*np.pi)
			point = center + 1.25*np.array([np.cos(angle), np.sin(angle)])
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
		self.G.node[3]['navs'] = list(range(6)) + [18]
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

		periph_secs = [0, 1, 4, 6, 5, 2]
		for idx, n in enumerate(periph_secs):
			self.G.add_edge(n, periph_secs[(idx+1)%len(periph_secs)])

		for i in range(6):
			self.G.G_nav.add_edge(18, i)
			self.G.G_nav.add_edge(i, (i+1)%6)
			self.G.G_nav.add_edge(i, 2*i+6)
			coin = (2*i+5) if (2*i+5)!=5 else 17
			self.G.G_nav.add_edge(i, coin)

		for i in range(6, 18):
			coin = i+1 if i+1 != 18 else 6
		 	self.G.G_nav.add_edge(i, coin)

		# print "TEST: Creating the following edges:"
		# print self.G.G_nav.edges()

	def do_voronoi(self):
		self.G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-0.5, 2.5))
		self.G.polygons = {i:pol for i, pol in self.G.polygons.items() if type(pol)==type(Polygon())}
		self.G.global_shape = cascaded_union(self.G.polygons.values())

	def show_network(self, secs=True, navs=True, show=True):
		if secs:
			plt.scatter(*zip(*[self.G.node[n]['coord'] for n in self.G.nodes()]), marker='s', color='r', s=50)

		if navs:
			plt.scatter(*zip(*[self.G.G_nav.node[n]['coord'] for n in self.G.G_nav.nodes()]), marker='o')
	
		if show:
			plt.show()

	def show_polygons(self, show=True):
		for pol in self.G.polygons.values():
			plt.fill(*zip(*list(pol.exterior.coords)), alpha=0.4)

		if show:
			plt.show()

class TestLowNetworkFunctions(SimpleSectorNetworkCase):
	def test_compute_voronoi(self):
		G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-1., 2.))
		
		coin = [list(pol.exterior.coords) for pol in G.polygons.values() if type(pol)==type(Polygon())]
		self.assertTrue(len(coin)==7)
		for pol in G.polygons.values():
			self.assertTrue(type(pol) == type(Polygon()))
		#for i, c in enumerate(coin):
		#	print i, c
		l = np.sqrt(3.)
		pouet = []
		center = np.array([0.5, l/2.])
		for i in range(6):
			angle = - np.pi/2.- float(i)/6.*2.*np.pi
			point = center + (1./l)*np.array([np.cos(angle), np.sin(angle)])
			pouet.append(list(point))
		pouet.append(pouet[0])
		self.assertTrue(len(coin[3])==7)
		self.assertTrue(assertCoordinatesAreEqual(coin[3], pouet))

	def test_reduce_airports_to_existing_nodes(self):
		airports = [0, 1, 10000]
		pairs = [(0, 1), (0, 100000), (10000, 1)]
		pairs, airports = reduce_airports_to_existing_nodes(self.G, pairs, airports)
		self.assertEqual(pairs, [(0, 1)])
		self.assertEqual(airports, [0, 1])

	def test_recompute_neighbors(self):
		G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-0.5, 2.5))
		G.add_edge(0, 6)
		recompute_neighbors(G)
		self.assertFalse(G.has_edge(0,6))

	def test_compute_navpoints_borders(self):
		borders_coordinates = [((0., 0.), (0., 1.))]
		shape = Polygon([(-1., -1.), (-1., 2.), (2., 2.), (2., -1.)])
		navpoints = compute_navpoints_borders(borders_coordinates, shape, lin_dens=1)   
		self.assertEqual(len(navpoints), 1)
		self.assertEqual(list(navpoints[0]), [10**(-5.), 0.5 + 10**(-5.)])

	def test_navpoints_at_borders(self):
		G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-0.5, np.sqrt(3.)+0.5))
		G.polygons = {i:pol for i, pol in G.polygons.items() if type(pol)==type(Polygon())}
		G.global_shape = cascaded_union(G.polygons.values())
		l = np.sqrt(3.)
		navpoints = navpoints_at_borders(G, lin_dens=1./l)

		self.assertEqual(len(navpoints), 22)

		# for nav in navpoints:
		# 	print nav

		# self.show_network(show=False)
		# self.show_polygons(show=False)
		# plt.scatter(*zip(*navpoints), marker='v', color='b')
		# plt.xlim((-1., 2.))
		# plt.ylim((-0.5, np.sqrt(3.)+0.5))
		# plt.show()

		witness = []
		center = np.array([0.5, l/2.])
		for i in range(6):
			angle = - np.pi/2. - np.pi/6. - float(i)/6.*2.*np.pi
			point = center + 0.5*np.array([np.cos(angle), np.sin(angle)])# + 10**(-5.)
			witness.append(list(point))

		navpoints_reduced = [nav for nav in navpoints if np.sqrt((nav[0]-center[0])**2 + (nav[1]-center[1])**2)<0.7]

		witness = [witness[i] for i in [0, 5, 1, 2, 3, 4]]

		self.assertEqual(len(navpoints_reduced), len(witness))
		self.assertTrue(assertCoordinatesAreEqual(navpoints_reduced, witness, thr=len(navpoints)*10**(-5.)))

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


		self.G = attach_two_sectors(3, 4, self.G)

		self.assertTrue(self.G.G_nav.has_edge(0, 6))

	def test_check_and_fix_empty_sectors(self):
		self.give_full_connections()
		self.give_nodes_to_secs()
		self.do_voronoi()
		#checked = {n:True for n in self.G.nodes()}
		
		G, problem = check_and_fix_empty_sectors(self.G, repair=False)

		self.assertFalse(problem)

		self.G.G_nav.remove_nodes_from([6, 17])
		self.G.node[4]['navs'] = []
		#checked[4] = False

		self.G, problem = check_and_fix_empty_sectors(self.G, repair=False)
		self.assertTrue(problem)
		self.G, problem = check_and_fix_empty_sectors(self.G, repair=True)
		self.assertTrue(problem)
		self.G, problem = check_and_fix_empty_sectors(self.G, repair=False)
		self.assertFalse(problem)

	def test_check_everybody_is_attached(self):
		self.give_full_connections()
		self.give_nodes_to_secs()

		self.G, problem = check_everybody_is_attached(self.G)
		self.assertFalse(problem)

		self.G.G_nav.remove_edge(16, 17)

		self.G, problem = check_everybody_is_attached(self.G)
		self.assertTrue(problem)
		self.G, problem = check_everybody_is_attached(self.G, repair=True)
		self.assertTrue(problem)
		self.G, problem = check_everybody_is_attached(self.G)
		self.assertFalse(problem)

	def test_check_everybody_has_one_cc(self):
		self.give_full_connections()
		self.give_nodes_to_secs()

		self.G, problem = check_everybody_has_one_cc(self.G, repair=False)
		self.assertFalse(problem)

		self.G.G_nav.remove_edges_from([(0, 5), (5, 18), (4, 18), (4, 3), (3, 18), (3, 2)])
		self.G.G_nav.node[4]['coord'] = np.array(self.G.G_nav.node[4]['coord']) + np.array([0., 0.01])
		# y component is to avoid interefering with 4 getting back with 18.
		self.G.G_nav.node[3]['coord'] = np.array(self.G.G_nav.node[3]['coord']) + np.array([0., 0.01])

		self.G, problem = check_everybody_has_one_cc(self.G, repair=False)
		self.assertTrue(problem)
		self.G, problem = check_everybody_has_one_cc(self.G, repair=True)
		self.assertTrue(problem)
		self.G, problem = check_everybody_has_one_cc(self.G, repair=False)
		self.assertFalse(problem)

		self.assertFalse(self.G.G_nav.has_edge(3, 18))
		self.assertTrue(self.G.G_nav.has_edge(4, 18))
		self.assertTrue(self.G.G_nav.has_edge(2, 3))
		self.assertFalse(self.G.G_nav.has_edge(5, 18))
		self.assertFalse(self.G.G_nav.has_edge(5, 0))
		self.assertFalse(self.G.G_nav.has_edge(4, 3))

	def test_check_matching(self):
		self.give_full_connections()
		self.give_nodes_to_secs()

		self.G, problem = check_matching(self.G, repair=False)
		self.assertFalse(problem)

		self.G.node[3]['navs'].remove(18)

		self.G, problem = check_matching(self.G, repair=False)
		self.assertTrue(problem)
		self.G, problem = check_matching(self.G, repair=True)
		self.assertTrue(problem)
		self.G, problem = check_matching(self.G, repair=False)
		self.assertFalse(problem)

		self.G.G_nav.node[18]['sec'] = 1
		self.G, problem = check_matching(self.G, repair=False)
		self.assertTrue(problem)
		self.G, problem = check_matching(self.G, repair=True)
		self.assertTrue(problem)
		self.G, problem = check_matching(self.G, repair=False)
		self.assertFalse(problem)

		del self.G.G_nav.node[18]['sec']
		with self.assertRaises(Exception):
			check_matching(self.G, repair=False)

	def test_compute_possible_outer_pairs(self):
		self.give_full_connections()
		self.give_nodes_to_secs()
		self.do_voronoi()

		navpoints = navpoints_at_borders(self.G, lin_dens=1./np.sqrt(3.))

		navpoints_idx = []
		nav_dic = {}
		for nav in navpoints:
			navpoints_idx.append(len(self.G.G_nav.nodes()))
			nav_dic[len(self.G.G_nav.nodes())] = nav
			self.G.G_nav.add_node(len(self.G.G_nav.nodes()), coord=nav)

		self.G.G_nav.navpoints_borders = navpoints_idx

		detect_nodes_on_boundaries({'make_borders_points':True}, self.G)

		self.G.G_nav.node[22]['sec'] = 0
		self.G.G_nav.node[23]['sec'] = 0
		self.G.G_nav.node[26]['sec'] = 1
		self.G.G_nav.node[27]['sec'] = 1
		self.G.G_nav.node[30]['sec'] = 2
		self.G.G_nav.node[35]['sec'] = 4
		self.G.G_nav.node[37]['sec'] = 5
		self.G.G_nav.node[38]['sec'] = 5
		self.G.G_nav.node[39]['sec'] = 6
		self.G.G_nav.node[40]['sec'] = 6

		pairs = compute_possible_outer_pairs(self.G)

		self.assertTrue([22, 35] in pairs)
		self.assertTrue([35, 22] in pairs)
		self.assertTrue([23, 35] in pairs)
		self.assertTrue([22, 37] in pairs)
		self.assertTrue([22, 38] in pairs)
		self.assertTrue([22, 39] in pairs)
		self.assertTrue([22, 40] in pairs)

		self.assertTrue(len(pairs)==4*(2*4 + 1*2) + 2*(2*2+ 1*1))

	def test_detect_nodes_on_boundaries(self):
		self.give_full_connections()
		self.give_nodes_to_secs()
		self.do_voronoi()

		navpoints = navpoints_at_borders(self.G, lin_dens=1./np.sqrt(3.))

		navpoints_idx = []
		for nav in navpoints:
			navpoints_idx.append(len(self.G.G_nav.nodes()))
			self.G.G_nav.add_node(len(self.G.G_nav.nodes()), coord=nav)

		self.G.G_nav.navpoints_borders = navpoints_idx

		detect_nodes_on_boundaries({'make_borders_points':True}, self.G)

		self.assertTrue(len(self.G.G_nav.outer_nodes)==10)

	def test_erase_small_sectors(self):
		self.give_full_connections()
		self.give_nodes_to_secs()

		erase_small_sectors(self.G, 0)

		self.assertTrue(len(self.G.nodes())==7)

		self.G.node[1]['navs'].remove(16)
		erase_small_sectors(self.G, 1)

		self.assertTrue(len(self.G.nodes())==6)
		self.assertFalse(self.G.has_node(1))

		erase_small_sectors(self.G, 2)
		self.assertTrue(len(self.G.nodes())==1)

	def test_find_pairs(self):
		self.give_full_connections()
		self.give_nodes_to_secs()

		all_airports = [9, 16, 11, 6, 15, 18]

		all_pairs = [(9, 16), (11, 6), (11, 15)]

		# Manual seed
		see_=1
		seed(see_)

		candidates, pairs = find_pairs(all_airports, all_pairs, 3, self.G)
		self.assertTrue(set(candidates)==set([11, 6, 15]))
		self.assertTrue(set(pairs)==set([(11, 6), (11, 15)]))
		
		candidates, pairs =	find_pairs(all_airports, all_pairs, 4, self.G)
		self.assertFalse(11 in candidates and 15 in candidates and 6 in candidates)

		candidates, pairs = find_pairs(all_airports, all_pairs, 2, self.G)
		self.assertTrue(set(candidates)==set([11, 6]) or set(candidates)==set([11, 15]))

		with self.assertRaises(Exception):
			find_pairs(all_airports, all_pairs, 6, self.G)
		
		#self.assertTrue(set(pairs)==set([(11, 6), (11, 15)]))

		#self.assertRaises(Pouet, find_pairs, all_airports, all_pairs, 4, self.G)


if __name__ == '__main__':
	#suite = unittest.TestLoader().loadTestsFromTestCase(TestLawNetworkFunctions)
	unittest.main(failfast=True)
