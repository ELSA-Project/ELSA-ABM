#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Choose and build the parameters for the creation of a navpoint network
superimposed with a sector network.
"""

from utilities import Paras
import pickle

def extract_data_from_files(paras_G):
	for item in ['net_sec', 'polygons', 'capacities', 'weights', 'airports', 'net_nav', 'flights_selected']:
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

			if item == 'airports':
				paras_G['nairports'] = len(paras_G['airports'])
		else:
			paras_G[item]=None

	# Check consistency of pairs and airports.
	for p1,p2 in paras_G['pairs']:
		try:
			assert p1 in paras_G['airports'] and p2 in paras_G['airports']
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
paras_G['name'] = ''				# This is just for info too.
if paras_G['file_net_sec'] == None:
	paras_G['N'] = 40 	# Number of sectors.
	paras_G['type_of_net'] = 'D' # 'D' for Delaunay, T for trianguler, E for Erdos-Renyi.


# ----------- Polygons ---------------
# Polygons corresponds to areas simulating the spatial extension of sectors.
# They are used to detect which sector each navpoint belongs to.
# Leave "None" for generation of new polygons.
# Input should be a pickle file with a dictionnary of polygons: name: Polygon object from shapely module.
paras_G['file_polygons'] = None


# ----------- Capacities ---------------
# Leave "None" for generation of new capacities.
# Input should be a pickle file with a dictionnay: node: capacity.
paras_G['file_capacities'] = None
#paras_G['file_capacities'] = 'capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'
paras_G['generate_capacities_from_traffic'] = False
if paras_G['file_capacities'] == None and not paras_G['generate_capacities_from_traffic']:
	paras_G['typ_capacities'] = 'constant'
	paras_G['C'] = 5                            # Sector capacity
	paras_G['C_airport'] = 20 					# Capacity of airports.
	paras_G['suppl_par_capacity']= None #additional parameters. See the desciption of Net.generate_capacities 
	#suppl_par_capacity_=['sqrt'] 
	#suppl_par_capacity = [0.3]	


# ------------- Weights --------------
# Here the weights are the times needed to cross edges. 
# Leave "None" for generation of new weights.
# Input should be a pickle file with a dictionary: edge: weight.

paras_G['file_weights'] = None
paras_G['generate_weights_from_traffic'] = False
if paras_G['file_capacities'] == None and not paras_G['generate_capacities_from_traffic']:
	paras_G['par_weights']= 3 # average crossing time for napvoint network in minutes.


# -------------- Traffic ---------------
# Leave "None" if you don't want to base neither weights nor capacities on traffic.
# Input should be a list of flight object as in the Distance library. TODO: description.
paras_G['file_flights_selected'] = None


# -------------- Airports -----------------
# Leave "None" for generation of new airports.

paras_G['file_airports'] = None
if paras_G['file_airports'] == None:
	paras_G['nairports'] = 2 	#number of airports, set to 0 for all airports (i.e. all sectors).
	# Manual selection of airports. Choose [] for random generation of airports with the previous number as parameter.
	#paras_G['airports']=[]#[65,20]      #IDs of the nodes used as airports
	#airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
	#airports=[65,22,10,45, 30, 16]
	#airports=[]
               
paras_G['pairs']=[]#[(22,65)]              #available connections between airports, set to [] for all possible connections (given min_dis).
#[65,20]
#[65,22]
#[65,62]
paras_G['min_dis']=5                       #minimum number of nodes (navpoints) between two airports (for a connection).


# ------------- Navpoints ----------------
# Leave "None" for generation of new navpoints.
paras_G['file_net_nav'] = None
if paras_G['file_net_nav'] == None: 
	paras_G['N_by_sectors']=10				# Number of navpoints per sector.


# --------------- Other ----------------
# Number of flights that the stratefic ABM will use. 
# It is the number of shortest paths computed for each pair of entry/exit
# Don't worry about this if you are not using the Strategic part of the ABM. 

paras_G['Nfp']=10		

# This is if you want to remove sectors with too few navpoints
paras_G['small_sec_thr'] = 4  # number of navpoints under which you remove the sector.	

# This is to transform all outer nodes (nodes on the outer boundary of the airspace)
paras_G['make_entry_exit_points'] = True			


# --------------- Distance ---------------- #
# This block is useful if you use the Distance library to select
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
		print "You asked to generate weights or capacities but did not provide traffic data."
		raise

# -------------- Read files --------------
# For now, only pickle files are supported.

paras_G = extract_data_from_files(paras_G)
