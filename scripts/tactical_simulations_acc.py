#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Script generating networks based on ACCs.
"""
import sys
sys.path.insert(1, '..')
import _mysql

from interface.abm_interface import do_ABM_tactical
from abm_strategic.interface_distance import name as name_net, paras_strategic
from libs.tools_airports import my_conv

def name_sim(name_G):
	return  'trajs_' + name_G

if __name__ == '__main__':
	airac = 334
	starting_date=[2010,5,6]
	n_days=1
	cut_alt=240.
	mode='navpoints'
	data_version=None
	n_iter = 10 # Does not control anything (TODO: change when interface with abm_tactical allows to control parameters)
	n_M1_trajs = 100 
	
	for country in ['LF', 'LE', 'EG', 'EB', 'LI']:
		paras = paras_strategic(zone=country, airac=airac, starting_date=starting_date, n_days=n_days, cut_alt=cut_alt,\
			mode=mode, data_version=data_version)

		db=_mysql.connect("localhost","root", paras['password_db'], "ElsaDB_A" + str(airac), conv=my_conv)

		db.query("""SELECT accName FROM ACCBoundary WHERE accName LIKE '""" + country + """%'""")
		r=db.store_result()
		rr=[rrr[0] for rrr in r.fetch_row(maxrows=0, how=0)]
		db.close()

		for zone in rr:
			print "=============================================="
			print "     Running abm: tactical for zone:", zone
			print "=============================================="

			for i in range(n_M1_trajs):
				paras_nav = paras_strategic(zone=zone, mode='navpoints', data_version=data_version)
				name_G = name_net(paras_nav, data_version)
				name_results = name_sim(name_G) + '_' + str(i)+ '.dat'
				#for j in range(n_iter):
				print '../trajectories/M3/' + name_results + '_' + str(i)
				do_ABM_tactical('../trajectories/M1/' + name_results, '../trajectories/M3/' + name_results)# + '_' + str(j))
