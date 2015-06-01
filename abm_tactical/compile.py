#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(1, '..')


def choose_paras(name_para, new_value, fil="../abm_tactical/config/config.cfg"):
	"""
	Function to modify a config file for the tactical ABM. If the given parameters does not 
	exist in the config file, the function does not modify anything and exit silently.

	Parameters
	----------
	name_para : string
		name of the parameter to update
	new_value : either string, float or integer
		new value of the parameter to update
	fil : string
		full path to the config file.

	Notes
	-----
	It is better to use this function with the help of the ParasTact class. 
	TODO: possibility of changing several parameters aa the same time.
	Note: This function is the same (or should be) than in abm_tactical/abm_interface.py 
	but it cannot be imported from it due to circular importations.

	"""

	with open(fil) as f:
		lines = f.readlines()
	#print "Trying to set", name_para, "to new value", new_value
	new_lines = []
	for i, l in enumerate(lines):
		if l[0]!='#' and len(l)>1: # If not a comment and not a blank line
			value, name = l.strip('\n').split('\t#')#split(l.strip(), '\t#')
			if name == name_para:
				#print "found", name_para, "I put new value", new_value
				line = str(new_value) + "\t#" + name + '\n'*(line[-1]=='\n') # last bit because of shock_f_lvl_min
			else:
				line = l[:]
		else:
			line = l[:]
		new_lines.append(line)

	with open(fil, 'w') as f:
		for line in new_lines:
			f.write(line)

if __name__ == '__main__':
	if len(sys.argv)==2 and sys.argv[1] == '--debug':
		#os.system("clang *.c -g3 -fsanitize=address -o ElsaABM.so -lm -shared -fno-omit-frame-pointer -fPIC -I/usr/include/python2.7/ -lpython2.7")
		#command ="clang *.c -g3 -fsanitize=address -o ElsaABM.so -lm "
		command = "clang mABM.c mQuery.c mSector.c main.c mTest.c mUtility.c -g3 -fsanitize=address -o ElsaABM.so -lm "
		os.system(command)
		print "Executed this command:", command
	else:
		command = "gcc -O3 mABM.c main.c mQuery.c mSector.c mTest.c mUtility.c -o ElsaABM.so -lm"
		os.system(command)
		print "Executed this command:", command
		#!/bin/bash
		command = "swig -python ABMtactic.i"
		os.system(command)
		print "Executed this command:", command
		command = "gcc -O3 -fpic -c -I/usr/include/python2.7 -I/usr/lib/python2.7/config/ *.c" 
		os.system(command)
		print "Executed this command:", command
		#gcc -shared main.o ABMtactic_wrap.o -lm -o _ABMtactic.so
		command = "gcc -O3 -shared *.o -lm -o _ABMtactic.so"
		os.system(command)
		print "Executed this command:", command

		maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		# print maindir
		choose_paras('main_dir', maindir) 
		#/home/earendil/Documents/ELSA/ABM/ABM_FINAL		#main_dir

		
	
