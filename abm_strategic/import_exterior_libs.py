#! /usr/bin/env python
# -*- coding: utf-8 -*-

#from shutil import copy
import os, sys
from string import split

"""
Just a method to copy automatically external libs.
"""

thisdir = os.path.dirname(__file__)

list_of_external_files = ['/home/earendil/Documents/ELSA/Modules/general_tools.py', 
							'/home/earendil/Documents/ELSA/Distance/tools_airports.py']
list_of_dir = [('/home/earendil/Programmes/YenKSP-master', 'YenKSP')]

if __name__== '__main__':
	for fil in list_of_external_files:
		#libdir = os.path.join(thisdir, '../' + fil)
		
		libdir = os.path.join(thisdir, '../libs/' + split(fil, "/")[-1])
		print "Copying lib from", fil
		os.system("cp " + fil + " " + libdir)
		if libdir not in sys.path:
			sys.path.insert(1, libdir)

	for rep, name in list_of_dir:
		libdir = os.path.join(thisdir, '../libs/' + name)
		print "Copying files from", rep
		os.system("mkdir -p " + libdir)
		os.system("cp -R " + rep + '/* ' + libdir + '/')
