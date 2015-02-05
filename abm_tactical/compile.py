#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(1, '..')

from interface.abm_interface import choose_paras

if __name__ == '__main__':
	if len(sys.argv)==2 and sys.argv[1] == '--debug':
		#os.system("clang *.c -g3 -fsanitize=address -o ElsaABM.so -lm -shared -fno-omit-frame-pointer -fPIC -I/usr/include/python2.7/ -lpython2.7")
		#command ="clang *.c -g3 -fsanitize=address -o ElsaABM.so -lm "
		command = "clang mABM.c mQuery.c mSector.c main.c mTest.c mUtility.c -g3 -fsanitize=address -o ElsaABM.so -lm "
		os.system(command)
		print "Executed this command:", command
	else:
		command = "gcc mABM.c main.c mQuery.c mSector.c mTest.c mUtility.c -o ElsaABM.so -lm"
		os.system(command)
		print "Executed this command:", command
		#!/bin/bash
		command = "swig -python ABMtactic.i"
		os.system(command)
		print "Executed this command:", command
		command = "gcc -fpic -c -I/usr/include/python2.7 -I/usr/lib/python2.7/config/ *.c" 
		os.system(command)
		print "Executed this command:", command
		#gcc -shared main.o ABMtactic_wrap.o -lm -o _ABMtactic.so
		command = "gcc -shared *.o -lm -o _ABMtactic.so"
		os.system(command)
		print "Executed this command:", command

		maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		print maindir
		choose_paras('main_dir', maindir) 
		#/home/earendil/Documents/ELSA/ABM/ABM_FINAL		#main_dir

		
	
