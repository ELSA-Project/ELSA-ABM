#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
import networkx as nx
import numpy as np
from random import seed

from abm_strategic.simAirSpaceO import *

################### Network Manager ########################

class AllocationTest(unittest.TestCase):
	def setUp(self):
		# Sectors
		self.G = Net()
		self.G.add_node(0, coord=(0., 0.))
		self.G.add_node(1, coord=(1., 0.))
		self.G.add_node(2, coord=(-0.5, np.sqrt(3.)/2.))
		self.G.add_node(3, coord=(0.5, np.sqrt(3.)/2.))
		self.G.add_node(4, coord=(1.5, np.sqrt(3.)/2.))
		self.G.add_node(5, coord=(0., np.sqrt(3.)))
		self.G.add_node(6, coord=(1., np.sqrt(3.)))

		# Sec-edges.
		self.G.add_edge(0, 1)
		self.G.add_edge(1, 4)
		self.G.add_edge(4, 6)
		self.G.add_edge(6, 5)
		self.G.add_edge(5, 2)
		#self.G.add_edge(2, 0)

		for i in [4, 5, 6]:
			self.G.add_edge(3, i)

		# Navigation Points
		self.G.G_nav = NavpointNet()
		l = np.sqrt(3.)

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

		# Connecting both network
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

		# Put nav-edges
		for i in range(6):
			self.G.G_nav.add_edge(18, i)
			self.G.G_nav.add_edge(i, (i+1)%6)
			self.G.G_nav.add_edge(i, 2*i+6)
			coin = (2*i+5) if (2*i+5)!=5 else 17
			self.G.G_nav.add_edge(i, coin)

		for i in range(6, 18):
			coin = i+1 if i+1 != 18 else 6
		 	self.G.G_nav.add_edge(i, coin)

		# Weights on nav-edges
		for n1, n2 in self.G.G_nav.edges():
			if not 18 in [n1, n2]:
				self.G.G_nav[n1][n2]['weight'] = 1.
			else:
				self.G.G_nav[n1][n2]['weight'] = 2.5
		self.G.G_nav.weighted = True

		# Remove some edges to create interesting paths.
		self.G.G_nav.remove_edges_from([(4, 13), (4, 14), (5, 15), (5, 16), (12, 13), (3, 12), (3, 11)])

		# Create fake shortest paths
		self.G.G_nav.short = {}
		self.G.G_nav.short[(14, 11)] = [[14, 15, 16, 17, 6, 7, 8, 9, 10, 11], [14, 15, 16, 17, 0, 18, 2, 10, 11]]
		self.G.G_nav.Nfp = 2
		self.G.Nfp = 2
		self.assertTrue(self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][0])<self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][1]))
		#print self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][0]), self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][1])
		self.G.airports = [0, 2]

	def put_capacities(self):
		for n in self.G.nodes():
			self.G.node[n]['capacity'] = 5
		for a in self.G.airports:
			self.G.node[a]['capacity_airport'] = 10000


class AirCompanyTest(AllocationTest):
	"""
	Tests also Flight
	"""
	
	def test_compute_flightplansR(self):
		# R company
		flight = Flight(0, 14, 11, 0., 0, (0., 0., 1.), 2)
		flight.compute_flightplans(5., self.G)
		
		self.assertTrue(len(flight.FPs)==2)
		
		paths = [fp.p_nav for fp in flight.FPs]
		times = [fp.t for fp in flight.FPs]

		self.assertTrue(paths == self.G.G_nav.short[(14, 11)])
		self.assertTrue(times == [0., 0.])

	def test_compute_flightplansS(self):
		# S company 
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)
		
		self.assertTrue(len(flight.FPs)==2)
		
		paths = [fp.p_nav for fp in flight.FPs]
		times = [fp.t for fp in flight.FPs]

		self.assertTrue(paths == 2*[self.G.G_nav.short[(14, 11)][0]])
		self.assertTrue(times == [0., 5.])

	def test_shift_desired_times(self):
		# S company
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)

		flight.shift_desired_time(20.)
		self.assertTrue([fp.t for fp in flight.FPs]==[20., 25.])

	def test_make_flags(self):
		# S company
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)

		for fp in flight.FPs:
			fp.accepted = False

		flight.make_flags()
		self.assertTrue(flight.flag_first==2)
		
	def test_fill_FPs(self):
		t0spV = [10.]
		tau = 5.

		# S company
		AC = AirCompany(10, 2, 1, [(14, 11)], (1., 0., 0.))
		AC.fill_FPs(t0spV, tau, self.G)

		self.assertTrue(len(AC.flights)==1)
		self.assertTrue(AC.flights[0].ac_id==10)
		self.assertTrue(AC.flights[0].pref_time==10.)

		paths = [fp.p_nav for fp in AC.flights[0].FPs]
		times = [fp.t for fp in AC.flights[0].FPs]

		self.assertTrue(paths == 2*[self.G.G_nav.short[(14, 11)][0]])
		self.assertTrue(times == [10., 15.])

class NetworkManagerTest(AllocationTest):
	# def __init__(self):
	# 	super(NetworkManagerTest, self).__init__()
	# 	# Change value of the wiehgts

	# 	for n1, n2 in self.G.G_nav.edges():
	# 		self.G.G_nav[n1][n2]['weights'] *= 10. 

	def setUp(self):
		super(NetworkManagerTest, self).setUp()

		for n1, n2 in self.G.G_nav.edges():
			self.G.G_nav[n1][n2]['weight'] *= 10. 

	def test_allocate_flight(self):
		self.put_capacities()

		NM = Network_Manager()
		NM.initialize_load(self.G)

		# S company 
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)
		
		self.assertTrue(len(flight.FPs)==2)
		
		paths = [fp.p_nav for fp in flight.FPs]
		times = [fp.t for fp in flight.FPs]

		self.assertTrue(paths == 2*[self.G.G_nav.short[(14, 11)][0]])
		self.assertTrue(times == [0., 5.])	

		NM.allocate_flight(self.G, flight, storymode=False)

		for n in self.G.nodes():
			if not n in [0, 1, 4, 6, 5, 2]:
				self.assertFalse(1 in self.G.node[n]['load'])

		for n in [0, 1, 4]:
			self.assertTrue(self.G.node[n]['load'][0]==1)
			self.assertTrue(self.G.node[n]['load'][1]==0)

		self.assertTrue(self.G.node[6]['load'][0]==1)
		self.assertTrue(self.G.node[6]['load'][1]==1)

		for n in [5, 2]:
			self.assertTrue(self.G.node[n]['load'][0]==0)
			self.assertTrue(self.G.node[n]['load'][1]==1)

	def test_compute_flight_times(self):
		NM = Network_Manager()
		NM.initialize_load(self.G)

		# S company 
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)

		NM.compute_flight_times(self.G, flight.FPs[0])

		self.assertTrue(flight.FPs[0].times==[0., 5., 25., 45., 65., 85.0, 90.])

	def test_overload_sector_hours(self):
		self.put_capacities()

		NM = Network_Manager()
		NM.initialize_load(self.G)

		self.G.node[0]['capacity'] = 5
		self.G.node[0]['load'] = [4] + 23*[0]

		self.assertFalse(NM.overload_sector_hours(self.G, 0, (0., 5.)))
		self.assertFalse(NM.overload_sector_hours(self.G, 0, (55., 65.)))
		self.assertFalse(NM.overload_sector_hours(self.G, 0, (0., 65.)))

		self.G.node[0]['capacity'] = 5
		self.G.node[0]['load'] = [5] + 23*[0]

		self.assertTrue(NM.overload_sector_hours(self.G, 0, (0., 5.)))
		self.assertTrue(NM.overload_sector_hours(self.G, 0, (55., 65.)))
		self.assertTrue(NM.overload_sector_hours(self.G, 0, (0., 65.)))

	def test_allocate_hours(self):
		self.put_capacities()

		NM = Network_Manager()
		NM.initialize_load(self.G)

		# S company 
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)

		NM.compute_flight_times(self.G, flight.FPs[0])

		NM.allocate_hours(self.G, flight.FPs[0])

		for n in [0, 1, 4]:
			self.assertTrue(self.G.node[n]['load'][0]==1)
			self.assertTrue(self.G.node[n]['load'][1]==0)

		self.assertTrue(self.G.node[6]['load'][0]==1)
		self.assertTrue(self.G.node[6]['load'][1]==1)

		for n in [5, 2]:
			self.assertTrue(self.G.node[n]['load'][0]==0)
			self.assertTrue(self.G.node[n]['load'][1]==1)

	def test_deallocate_hours(self):
		self.put_capacities()

		NM = Network_Manager()
		NM.initialize_load(self.G)

		# S company 
		flight = Flight(0, 14, 11, 0., 0, (1., 0., 0.), 2)
		flight.compute_flightplans(5., self.G)

		NM.compute_flight_times(self.G, flight.FPs[0])

		NM.allocate_hours(self.G, flight.FPs[0])
		NM.deallocate_hours(self.G, flight.FPs[0])

		for n in self.G.nodes():
			for p in self.G.node[n]['load']:
				self.assertTrue(p==0)

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
		pass

if __name__ == '__main__':
	unittest.main(failfast=True)
