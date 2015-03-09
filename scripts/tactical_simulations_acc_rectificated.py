#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Script generating networks based on ACCs.
"""
import sys, os
sys.path.insert(1, '..')
import _mysql
import pickle
from os.path import join as jn

from paths import result_dir
from interface.abm_interface import do_ABM_tactical, choose_paras
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

	main_dir = os.path.abspath(__file__)
	main_dir = os.path.split(os.path.dirname(main_dir))[0]

	dontdo = ['EGCC', 'EGPX']
	
	for country in ['LI']:#, 'EB', 'LI']:
		paras = paras_strategic(zone=country, airac=airac, starting_date=starting_date, n_days=n_days, cut_alt=cut_alt,\
			mode=mode, data_version=data_version)

		db=_mysql.connect("localhost","root", paras['password_db'], "ElsaDB_A" + str(airac), conv=my_conv)

		db.query("""SELECT accName FROM ACCBoundary WHERE accName LIKE '""" + country + """%'""")
		r=db.store_result()
		rr=[rrr[0] for rrr in r.fetch_row(maxrows=0, how=0)]
		db.close()

		for zone in rr:
			if not zone in dontdo:
				print "=============================================="
				print "     Running abm: tactical for zone:", zone
				print "=============================================="
				with open('../libs/All_shapes_334.pic','r') as f:
					all_shapes = pickle.load(f)
				boundary = list(all_shapes[zone]['boundary'][0].exterior.coords)
				assert boundary[0]==boundary[-1]

				with open('../abm_tactical/config/bound_latlon.dat', 'w') as f:
					for x, y in boundary:
						f.write(str(x) + '\t' + str(y) + '\n')

				for i in range(n_M1_trajs):
					if i==0:
						# Compute temporary points only for first iteration
						choose_paras('tmp_from_file', 0)
					else:
						choose_paras('tmp_from_file', 1)
					paras_nav = paras_strategic(zone=zone, mode='navpoints', data_version=data_version)
					name_G = name_net(paras_nav, data_version)
					name_results = name_sim(name_G) + '_' + str(i)+ '.dat'
					#for j in range(n_iter):
					#print '../trajectories/M3/' + name_results + '_' + str(i)
					input_name = jn(result_dir, 'trajectories/M1/' + name_results)
					output_name = jn(result_dir, 'trajectories/M3/' + name_results)
					do_ABM_tactical(input_name, output_name)# + '_' + str(j))
