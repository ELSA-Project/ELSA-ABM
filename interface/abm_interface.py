#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
====================================================================
--------------------------- Interface ------------------------------
====================================================================
This file provides python functions to easily use the tactical abm
within a python script.
"""

import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic')

import os
from string import split
import numpy as np
import pickle
import random as rd
from os.path import join as jn

from abm_tactical import *
from abm_strategic import *

from libs.paths import main_dir
from libs.general_tools import silence

if 0:
    # Manual seed
    see_=1
    #see_ = 15 #==> the one I used for new, old_good, etc.. 
    # see=2
    print "===================================="
    print "          USING SEED", see_
    print "===================================="
    np.random.seed(see_)

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

def do_ABM_tactical(input_file, output_file, config_file, verbose=2, 
	shock_tmp=jn(main_dir, 'abm_tactical/config/shock_tmp.dat'),
	bound_latlon=jn(main_dir, 'abm_tactical/config/bound_latlon.dat'),
	temp_nvp=jn(main_dir, 'abm_tactical/config/temp_nvp.dat')):

	"""
	Main function of control of the tactical ABM. The function uses tactical_simulation,
	which has beem compiled precedently using the wrapper.

	Parameters
	----------
	input_file : string
		full path to M1 file (planned trajectories)
	output_file : string
		full path to save M3 file (trajectories after control)
	config_file : string
		full path to config file.
	verbose : integer, optional
		verbosity
	shock_tmp : string, optional
		full path to file containing the possible coordinates of shocks.
	bound_latlon : string, optional
		full path to file containing the coordinates of the boundary of the controlled airspace.
	temp_nvp : string
		full path to file containing temporary navigation points used in the simulations.
	"""

	for fil, name in [(shock_tmp, 'shock_tmp'), (bound_latlon, 'bound_latlon'), (temp_nvp, 'temp_nvp')]:
		choose_paras(name, fil, fil=config_file)

	try:
		inpt = ["", input_file, output_file, config_file]

		if verbose>1:
			print "M1 source:", inpt[1]
			print "Destination output:", inpt[2]
			print "Config file:", inpt[3]
			print
			print "Running ABM Tactical model..."

		with silence(verbose==0): # Does not work.
			tactical_simulation(inpt)
	except:
		pass #TODO	
	if verbose==2:
		print
		print "Done."

class ParasTact(Paras):
	"""
	Class
	=====
	This is class used to handle a config file of the tactical ABM. It is used 
	to update the value of some parameters. The method initialize_paras must be called once 
	before the beginning of the simulations.
	"""

	def update(self, name_para, value_para, config_file=None):
		super(ParasTact, self).update(name_para, value_para)
		for k in self.to_update.keys() + self['paras_to_loop']:
			choose_paras(k, self[k], fil=config_file)

	def initialize_paras(self, config_file=None):
		"""
		This is here only because the update procedure es not
		compute the paras derived from other paras (via to_update)
		if the arguments of the functions have not changed. This is
		problematic for initialization because the derived paras
		might not be present in the dictionnary until one of their 
		arguments has changed.
		"""

		for key in self['paras_to_loop']:
			self[key] = self[key + '_iter'][0]

		keys = self.update_priority
		for key in keys:
			f, args = self.to_update[key]
			self[key] = f(*[self[arg] for arg in args])

		for key in self.keys():
			choose_paras(key, self[key], fil=config_file)

if __name__ == '__main__':
	"""
	Manual entry.
	"""

	input_file = jn(result_dir, "trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat")
	output_file = jn(result_dir, "results/output.dat")
	config_file = jn(main_dir, 'abm_tactical/config/config.cfg')
	shock_tmp = jn(main_dir, 'abm_tactical/config/shock_tmp.dat')
	bound_latlon = jn(main_dir, 'abm_tactical/config/bound_latlon.dat')
	print bound_latlon
	temp_nvp = jn(main_dir, 'abm_tactical/config/temp_nvp.dat')


	choose_paras('tmp_from_file', 1)
	#input_file = os.path.join(main_dir, "trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat")
	#input_file = os.path.join(result_dir, "trajectories/M1/trajs_Real_LF_v5.8_Strong_EXTLFBB_LFBB_2010-5-6+0_d2_cut240.0_directed_1.dat")
	#output_file = os.path.join(main_dir, "results/output.dat")
	#output_file = os.path.join(result_dir, "trajectories/M3/trajs_Real_LF_v5.8_Strong_EXTLFBB_LFBB_2010-5-6+0_d2_cut240.0_directed_1.dat")
	#do_ABM_tactical(input_file, output_file)
	
	do_ABM_tactical(input_file, output_file, config_file, verbose=2, 
									shock_tmp=shock_tmp,
									bound_latlon=bound_latlon,
									temp_nvp=temp_nvp)


