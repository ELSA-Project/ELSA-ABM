#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: earendil

Script generating networks based on ACCs.
"""
import sys
sys.path.insert(1, '..')
import _mysql

from abm_strategic.interface_distance import paras_strategic, build_net_distance
from abm_strategic.prepare_navpoint_network import NoEdges
from libs.tools_airports import my_conv

if __name__=='__main__':
	airac = 334
	starting_date=[2010,5,6]
	n_days=1
	cut_alt=240.
	mode='navpoints'
	data_version=None
	
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
			print "Building network for zone:", zone
			print "=============================================="
			try:
				build_net_distance(zone=zone, show=False)
			except NoEdges:
				print "This ACC has only one sector, I skip it."
			print 
			print 
		