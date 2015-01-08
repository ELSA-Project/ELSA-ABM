#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Script generating networks based on ACCs.
"""
import sys
sys.path.insert(1, '..')
import _mysql

from abm_strategic.simulationO import generate_traffic
from abm_strategic.interface_distance import name as name_net, paras_strategic
from abm_strategic.utilities import network_whose_name_is
from libs.tools_airports import my_conv
from libs.general_tools import silence, counter

def name_sim(name_G):
	return  'trajs_' + name_G


if __name__=='__main__':
	airac = 334
	starting_date=[2010,5,6]
	n_days=1
	cut_alt=240.
	mode='navpoints'
	data_version=None
	n_iter = 10
	
	for country in ['LF', 'LE', 'EG', 'EB', 'LI']:
		paras = paras_strategic(zone=country, airac=airac, starting_date=starting_date, n_days=n_days, cut_alt=cut_alt,\
			mode=mode, data_version=data_version)

		db=_mysql.connect("localhost","root", paras['password_db'], "ElsaDB_A" + str(airac), conv=my_conv)

		db.query("""SELECT accName FROM ACCBoundary WHERE accName LIKE '""" + country + """%'""")
		r=db.store_result()
		rr=[rrr[0] for rrr in r.fetch_row(maxrows=0,how=0)]
		db.close()

		for zone in rr:
			print "=============================================="
			print "Generating traffic for zone:", zone
			print "=============================================="
			paras_nav = paras_strategic(zone=zone, mode='navpoints', data_version=data_version)
			name_G = name_net(paras_nav, data_version)
			try:
				G = network_whose_name_is('../networks/' + name_G)
			except IOError:
				print "Could not load the network, I continue."
				print 
				continue

			for i in range(n_iter):
				counter(i, n_iter, message="Doing simulations...")
				name_results = name_sim(name_G) + '_' + str(i)
				with silence(True):
					generate_traffic(G, save_file='../trajectories/' + name_results, file_traffic="../networks/" + name_G + '_flights_selected.pic')
		
			print 
			print 