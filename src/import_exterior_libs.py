#! /usr/bin/env python
# -*- coding: utf-8 -*-

#from shutil import copy
import os

"""
Just a method to copy automatically some other custom libs.
"""

list_of_external_files = ['/home/earendil/Documents/ELSA/Modules/general_tools.py', 
							'/home/earendil/Documents/ELSA/Distance/tools_airports.py']

def import_ext_libs():
	for f in list_of_external_files:
		#copy(f , '.')
		os.system("cp " + f + " .")