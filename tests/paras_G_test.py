#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for the construction of the parameter dictionary for the creation of a sector network
superimposed with a navpoint network (hybrid network).
"""

from utilities import Paras
import pickle

def extract_data_from_files(paras_G):
	for item in ['net_sec', 'polygons', 'capacities', 'weights', 'airports_sec', 'airports_nav', 'net_nav', 'flights_selected']:
		if paras_G['file_' + item]!=None:
			try:
				f = open( paras_G['file_' + item])
				try:
					paras_G[item] = pickle.load(f)
					print "Loaded file", paras_G['file_' + item], "for", item
				except:
					print "Could not load the file", paras_G['file_' + item], " as a pickle file."
					print "I skipped it and the", item, "will be generated or ignored."
					pass # TODO
				f.close()
			except:
				print "Could not find file",  paras_G['file_' + item]
				print "I skipped it and the", item, "will be generated or ignored."
				pass

			if item == 'airports_sec':
				paras_G['nairports_sec'] = len(paras_G['airports_sec'])
			if item == 'airports_nav':
				paras_G['nairports_nav'] = len(paras_G['airports_nav'])
		else:
			paras_G[item]=None

	# Check consistency of pairs and airports.
	for p1, p2 in paras_G['pairs_nav']:
		try:
			assert p1 in paras_G['airports_nav'] and p2 in paras_G['airports_nav']
		except:
			print "You asked a connection for which one of the airport does not exist:", (p1, p2)
			raise
	for p1, p2 in paras_G['pairs_sec']:
		try:
			assert p1 in paras_G['airports_sec'] and p2 in paras_G['airports_sec']
		except:
			print "You asked a connection for which one of the airport does not exist:", (p1, p2)
			raise

	return paras_G
#from tools_airports import get_paras, extract_flows_from_data

paras_G = {}

# ----------- Network of sector ------------
# Leave "None" for generation of a new network
# Input should be a pickle file with a networkx object.

paras_G['file_net_sec'] = None
paras_G['type_of_net'] = ''			# This is just for information (e.g. name of the file).
paras_G['name'] = 'Example'			# This is just for info too.
if paras_G['file_net_sec'] == None:
	paras_G['N'] = 10 	# Number of sectors.
	paras_G['type_of_net'] = 'D' # 'D' for Delaunay, T for trianguler, E for Erdos-Renyi.


# ----------- Polygons ---------------
# Polygons correspond to areas simulating the spatial extension of sectors (in 2d).
# They are used to detect which sector each navpoint belongs to.
# Leave "None" for generation of new polygons using Voronoi tessellation.
# Input should be a pickle file with a dictionnary of polygons: node (sector): Polygon object from shapely module.
paras_G['file_polygons'] = None


# -------------- Traffic ---------------
# Leave "None" if you don't want to base neither weights nor capacities on traffic.
# Input should be a list of flight object as in the Distance library. TODO: description.
paras_G['file_flights_selected'] = None


# ----------- Capacities ---------------
# Leave "None" for generation of new capacities.
# Input should be a pickle file with a dictionnay: node (sector): capacity.
paras_G['file_capacities'] = None
#paras_G['file_capacities'] = 'capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'
paras_G['generate_capacities_from_traffic'] = False
if paras_G['file_capacities'] == None and not paras_G['generate_capacities_from_traffic']:
	paras_G['typ_capacities'] = 'constant'
	paras_G['C'] = 5                            # Sector capacity (flights per hour)
	paras_G['C_airport'] = 1000					# Capacity of airports.
	paras_G['suppl_par_capacity']= None #additional parameters. See the desciption of Net.generate_capacities 
	#suppl_par_capacity_=['sqrt'] 
	#suppl_par_capacity = [0.3]	


# -------------- Airports -----------------
# Leave "None" for generation of new airports.
# Be careful with this, because you need specify explicitly the airports (entry/exits of the sector network)
# and the entry/exits for the network navpoint, they have to be consisten, i.e. all sectors containing
# a navpoint which is entry/exit must be an airport. Generally, it is preferred to set navpoints entry/exits
# this section as it is.

# WARNING: you should not change these parameters.
paras_G['file_airports_sec'] = None
if paras_G['file_airports_sec'] == None:
	paras_G['airports_sec'] = None
	if paras_G['airports_sec'] == None:
		paras_G['nairports_sec'] = 10
paras_G['pairs_sec'] = [] 	# List of possible connections between airports. Leave [] to 
							# to select all possible connections.

# ------------- Navpoint network ----------------
# Leave "None" for generation of new navpoints.
paras_G['file_net_nav'] = None
if paras_G['file_net_nav'] == None: 
	paras_G['N_by_sectors'] = 10				# Number of navpoints per sector.


# ------------- Weights --------------
# Here the weights are the times needed to cross edges (between navpoints). 
# Leave "None" for generation of new weights.
# Input should be a pickle file with a dictionary: edge: weight.

paras_G['file_weights'] = None
paras_G['generate_weights_from_traffic'] = False
if paras_G['file_weights'] == None and not paras_G['generate_weights_from_traffic']:
	paras_G['par_weights'] = 3 # average crossing time for napvoint network in minutes.


# -------------- Entry/Exits -----------------
# The ``airports'' for the navpoints are just entry/exits of the airspace. One can specify either
# a file, a list of label of nodes, or just a number of nodes, to be d

paras_G['file_airports_nav'] = None
if paras_G['file_airports_nav'] == None:
	paras_G['airports_nav'] = None
	if paras_G['airports_nav'] == None:
		paras_G['nairports_nav'] = 5

paras_G['function_airports_nav'] = None #TODO
paras_G['pairs_nav'] = [] 	# List of possible connections between entries and exits. Leave [] to 
							# to select all possible connections.
paras_G['min_dis'] = 5      # minimum number of nodes (navpoints) between entry and exit.

# If True, keep singletons, i.e. the same sec-aiport in entry and exit. Otherwise remove
# pairs of nav-entry/exits which yield the same sec-airport in entrey and exit.
paras_G['singletons'] = False

# If True, detects the outer nodes  and set them as airports. 
# All previous parameters are discarded for entry/exits.
paras_G['use_only_outer_nodes_for_airports'] = False


# --------------- Other Parmaters ----------------

# Number of flights that the stratefic ABM will use. 
# It is the number of shortest paths computed for each pair of entry/exit
# Don't worry about this if you are not using the Strategic part of the ABM. 
paras_G['Nfp'] = 3	

# This is if you want to remove sectors with too few navpoints
paras_G['small_sec_thr'] = 0  # number of navpoints under which you remove the sector. 0: no threshold.

# This is to create navpoints on each border of sector. Note that even False will trigger the
# creation of border points on the outer boundary (TODO: Change this?).
paras_G['make_borders_points'] = False

# Linear density of points on the borders
paras_G['lin_dens_borders'] = 5

# Attach non-airport nodes with degree 1 to closest neighbor.
paras_G['attach_termination_nodes'] = True


# Expand polygons outwards in case all points are not detected to be in one of them.
# The value is the approximate fraction of maximum expansion (0 for no expansion).
# The function "expand" is in libs.tools_airports.
# Put 1. if you don't want to expand the sectors.
paras_G['expansion'] = 1.

# --------------- Distance ---------------- #
# This block is useful if you use the Distance library (based on the ELSA database) to select
# sectors, navpoints, polygons, or traffic data
#layer_ = 350

# paras_real_ = get_paras()
# paras_real_['zone'] = country_
# paras_real_['airac'] = airac_
# paras_real_['type_zone'] ='EXT'
# paras_real_['filtre'] ='Weak'
# paras_real_['mode'] ='navpoints'
# paras_real_['cut_alt'] = 240.
# paras_real_['both'] = False
# paras_real_['n_days'] = 1

# ----------- Checks -------------

if paras_G['generate_weights_from_traffic'] or paras_G['generate_capacities_from_traffic']:
	try:
		assert paras_G['flights_selected'] != None
	except:
		raise Exception("You asked to generate weights or capacities but did not provide traffic data.")

# -------------- Read files --------------
# For now, only pickle files are supported.
# TODO: automatically detect and read csv files.

paras_G = extract_data_from_files(paras_G)
