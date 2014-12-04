#! /usr/bin/env python
# -*- coding: utf-8 -*-

#from shutil import copy
import os, sys
from string import split

"""
Just a method to copy automatically some other custom libs.
"""

thisdir = os.path.dirname(__file__)

list_of_external_files = ['/home/earendil/Documents/ELSA/Modules/general_tools.py', 
							'/home/earendil/Documents/ELSA/Distance/tools_airports.py']

# def import_ext_libs(where ='.'):
# 	for f in list_of_external_files:
# 		#copy(f , '.')
# 		os.system("cp " + f + " " + where)

if len(sys.argv)>1 and sys.argv[1] == 'b':
	for fil in list_of_external_files:
		#libdir = os.path.join(thisdir, '../' + fil)

		libdir = os.path.join(thisdir, '../libs/' + split(fil, "/")[-1])

		os.system("cp " + fil + " " + libdir)
		if libdir not in sys.path:
			sys.path.insert(1, libdir)