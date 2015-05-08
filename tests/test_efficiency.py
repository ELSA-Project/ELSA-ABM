#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from libs.efficiency import *
from abm_strategic.simAirSpaceO import Net, NavpointNet 
from abm_strategic.prepare_navpoint_network import compute_voronoi


class RectificationNetworkTest(unittest.TestCase):
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
		 #self.G.G_nav.remove_edges_from([(4, 13), (4, 14), (5, 15), (5, 16), (12, 13), (3, 12), (3, 11)])

	def do_voronoi(self):
		self.G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-0.5, 2.5))
		self.G.polygons = {i:pol for i, pol in self.G.polygons.items() if type(pol)==type(Polygon())}
		self.G.global_shape = cascaded_union(self.G.polygons.values())

	def test_rectificate_no_resampling(self):
		trajs = [[0, 1, 2, 3, 4, 5]]
		eff_target = 1.

		trajs_rec, eff, G, groups_rec = rectificate_trajectories_network(trajs, eff_target,	self.G,	remove_nodes=True, 
																									resample_trajectories=False)
		self.assertTrue(len(trajs_rec)==1)
		self.assertTrue(trajs_rec[0]==[0, 5])

	def test_rectificate_resampling(self):
		self.do_voronoi()
		trajs = [[0, 1, 2, 3, 4, 5]]
		eff_target = 1.

		trajs_rec, eff, G, groups_rec = rectificate_trajectories_network(trajs, eff_target,	self.G,	remove_nodes=True, 
																									resample_trajectories=True)
		self.assertTrue(len(trajs_rec)==1)
		self.assertTrue(len(trajs_rec[0])==6)
		self.assertTrue(trajs_rec[0][0]==0)
		self.assertTrue(trajs_rec[0][-1]==5)

		# Check if all resampled points are at the same distance from each other.
		prev_dis = (self.G.G_nav.node[0]['coord'][0] - self.G.G_nav.node[1]['coord'][0])**2 + (self.G.G_nav.node[0]['coord'][1] - self.G.G_nav.node[1]['coord'][1])**2
		for i in range(1, len(trajs_rec[0])-1):
			dis = (self.G.G_nav.node[i]['coord'][0] - self.G.G_nav.node[i+1]['coord'][0])**2 + (self.G.G_nav.node[i]['coord'][1] - self.G.G_nav.node[i+1]['coord'][1])**2
			self.assertTrue(abs(dis-prev_dis)<10**(-5.))
			prev_dis = dis

		# Check if the new nodes are in the right sector
		print [G.G_nav.node[n]['sec'] for n in G.G_nav.nodes() if n>18]

	def test_find_sector(self):
		self.do_voronoi()
		self.assertTrue(find_sector(18, self.G)==3)
		self.assertTrue(find_sector(7, self.G)==6)


if __name__ == '__main__':
	# Put failfast=True for stopping the test as soon as one test fails.
	unittest.main(failfast=True)
	

