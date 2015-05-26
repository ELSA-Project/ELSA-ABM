#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
import os

from abm_strategic.simAirSpaceO import Net, NavpointNet, Flight
from abm_strategic.simulationO import *

# Next ones to do: test on add_first_last_points

# TODO: high level tests

class SimulationTest(unittest.TestCase):
	def setUp(self):
		self.prepare_network()

	def prepare_network(self):
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
		self.G.Nsp_nav = 1
		self.G.Nfp = 2
		self.assertTrue(self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][0])<self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][1]))
		#print self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][0]), self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][1])
		self.G.airports = [0, 2]

		for n in self.G.nodes():
			self.G.node[n]['capacity'] = 5
		for a in self.G.airports:
			self.G.node[a]['capacity_airport'] = 10000

		self.G.comments = []


class FunctionsTest(SimulationTest):
	def test_build_path(self):
		self.paras = {}

		# Fo S Companies
		self.paras['par'] = [(1., 0., 0.), (1., 0., 0.)]
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'

		class Pouet:
			pass

		self.paras['G'] = Pouet()
		self.paras['G'].name = 'GNAME'

		coin = build_path(self.paras, vers='0.0', in_title=[], rep='')

		self.assertTrue(coin=='Sim_v0.0_GNAME')

	def test_post_process_queue(self):
		queue = []
		queue.append(Flight(0, 14, 11, 0., 0, (1., 0., 1.), 2))
		queue.append(Flight(1, 14, 11, 0., 1, (1., 0., 1.), 2))

		for f in queue:
			f.compute_flightplans(20., self.G)
			f.best_fp_cost=f.FPs[0].cost

		queue[1].FPs[0].accepted = False
		queue[1].FPs[1].accepted = True

		queue = post_process_queue(queue)

		self.assertTrue(queue[0].satisfaction==1.)
		self.assertTrue(queue[0].regulated_1FP==0.)
		self.assertTrue(queue[0].regulated_FPs==0)
		self.assertTrue(queue[0].regulated_F==0.)

		self.assertTrue(queue[1].satisfaction==9./11.)
		self.assertTrue(queue[1].regulated_1FP==1.)
		self.assertTrue(queue[1].regulated_FPs==1)
		self.assertTrue(queue[1].regulated_F==0.)

	def test_add_first_last_points(self):
		trajs = []
		trajs.append([(0., 0., 0, [2010, 1, 1, 0, 0, 0]), (0., 1., 0, [2010, 1, 1, 0, 10, 0]), (0., 4., 0, [2010, 1, 1, 0, 30, 0])])

		trajs_modified = add_first_last_points(trajs, dummy_sec=None)

		self.assertTrue(len(trajs_modified[0])==5)
		for point in trajs_modified[0]:
			self.assertTrue(len(point)==4)
		self.assertTrue(trajs_modified[0][0]==(0., -1., 0, [2009, 12, 31, 23, 50, 0]))
		self.assertTrue(trajs_modified[0][-1]==(0., 7., 0, [2010, 1, 1, 0, 50, 0]))

		# with sectors
		trajs = []
		trajs.append([(0., 0., 0, [2010, 1, 1, 0, 0, 0], 5), (0., 1., 0, [2010, 1, 1, 0, 10, 0], 5), (0., 4., 0, [2010, 1, 1, 0, 30, 0], 5)])

		trajs_modified = add_first_last_points(trajs, dummy_sec=-10)

		self.assertTrue(len(trajs_modified[0])==5)
		for point in trajs_modified[0]:
			self.assertTrue(len(point)==5)
		self.assertTrue(trajs_modified[0][0]==(0., -1., 0, [2009, 12, 31, 23, 50, 0], -10))
		self.assertTrue(trajs_modified[0][-1]==(0., 7., 0, [2010, 1, 1, 0, 50, 0], -10))
		

class SimulationTimesZeros(SimulationTest):
	def setUp(self):
		super(SimulationTimesZeros, self).setUp()
		self.prepare_paras()

	def prepare_paras(self):
		self.paras = {}

		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)] # For S and R Companies
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'
		self.paras['ACtot'] = 2
		self.paras['na'] = 1
		self.paras['G'] = self.G
		self.paras['tau'] = 20.
		self.paras['nA'] = 1
		self.paras['Nsp_nav'] = 1
		self.paras['old_style_allocation'] = False
		self.paras['noise'] = 0.
		self.paras['AC'] = 2

	def test_initialization(self):
		sim = Simulation(self.paras, G=self.G, verbose=False)

		self.assertTrue(len(sim.t0sp)==self.paras['ACtot'])
		for p in sim.t0sp:
			self.assertTrue(len(p)==1)
			self.assertTrue(p[0]==0.)

	def test_build_ACs(self):
		self.paras['AC'] = 2
		self.paras['par'] = [(1., 0., 0.)]
		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs()

		self.assertTrue(len(sim.ACs)==self.paras['AC'])
		
		for ac in sim.ACs.values():
			self.assertTrue(ac.par==(1., 0., 0.))
			self.assertTrue(ac.flights[0].pref_time==0.)

	def test_build_ACs2(self):
		self.paras['AC'] = 2
		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)]
		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs()

		self.assertTrue(len(sim.ACs)==self.paras['AC'])
		
		self.assertTrue(sim.ACs[0].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[1].par==(0., 0., 1.))

	def test_build_ACs3(self):
		self.paras['AC'] = [2, 0]
		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)]
		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs()

		self.assertTrue(len(sim.ACs)==sum(self.paras['AC']))
		
		self.assertTrue(sim.ACs[0].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[1].par==(1., 0., 0.))


class SimulationFlows(SimulationTest):
	def setUp(self):
		super(SimulationFlows, self).setUp()
		self.prepare_paras()
		self.finish_network()

	def finish_network(self):
		self.G.G_nav.idx_nodes = {14:14, 11:11}
		self.G.G_nav.airports = [11, 14]
		self.G.G_nav.short[(11, 14)] = [[11, 10, 9, 8, 7, 6, 17, 16, 15, 14], [11, 10, 2, 18, 0, 17, 16, 15, 14]]
		
	def prepare_paras(self):
		self.paras = {}

		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)] # For S and R Companies
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'
		self.paras['ACtot'] = 4
		self.paras['na'] = 1
		self.paras['G'] = self.G
		self.paras['tau'] = 20.
		self.paras['nA'] = 1
		self.paras['Nsp_nav'] = 1
		self.paras['old_style_allocation'] = False
		self.paras['noise'] = 0.
		#self.paras['AC'] = 2

	def test_build_ACs_from_flows(self):
		self.paras['flows'] = {}
		self.paras['flows'][(14, 11)] = [[2010, 1, 1, 0, 0, 0], [2010, 1, 1, 0, 10, 0]]
		self.paras['flows'][(11, 14)] = [[2010, 1, 1, 0, 20, 0], [2010, 1, 1, 0, 30, 0]]
		self.paras['bootstrap_mode'] = False
		self.paras['nA'] = 0.5
		self.paras['ACtot'] = 4

		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs_from_flows()

		self.assertTrue(sim.starting_date==[2010, 1, 1, 0, 0, 0])

		self.assertTrue(len(sim.ACs)==self.paras['ACtot'])

		self.assertTrue(sim.ACs[0].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[1].par==(0., 0., 1.))
		self.assertTrue(sim.ACs[2].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[3].par==(0., 0., 1.))
	

		self.assertTrue(sim.ACs[0].flights[0].pref_time==0.)
		self.assertTrue(sim.ACs[1].flights[0].pref_time==10.)
		self.assertTrue(sim.ACs[2].flights[0].pref_time==20.)
		self.assertTrue(sim.ACs[3].flights[0].pref_time==30.)

	def test_build_ACs_from_flows_bootstrap(self):
		self.paras['flows'] = {}
		self.paras['flows'][(14, 11)] = [[2010, 1, 1, 0, 0, 0], [2010, 1, 1, 0, 10, 0]]
		self.paras['flows'][(11, 14)] = [[2010, 1, 1, 0, 20, 0], [2010, 1, 1, 0, 30, 0]]
		self.paras['bootstrap_mode'] = True
		self.paras['bootstrap_only_time'] = True
		self.paras['nA'] = 1.
		self.paras['ACtot'] = 6

		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs_from_flows()

		self.assertTrue(sim.starting_date==[2010, 1, 1, 0, 0, 0])

		self.assertTrue(len(sim.ACs)==self.paras['ACtot'])

		for ac in sim.ACs.values():
			self.assertTrue(ac.flights[0].pref_time in [0., 10., 20., 30.])

		#print [ac.flights[0].pref_time for ac in sim.ACs.values()]


if __name__ == '__main__':
	# Manual tests
	os.system('../abm_strategic/simulationO.py paras_test.py')
	#os.system('../abm_strategic/iter_simO.py paras_iter_test.py')
	
	# Put failfast=True for stopping the test as soon as one test fails.
	unittest.main(failfast=True)



