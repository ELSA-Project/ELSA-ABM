#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

if __name__ == '__main__':
	if len(sys.argv)==2 and sys.argv[1] == '--debug':
		#os.system("clang *.c -g3 -fsanitize=address -o ElsaABM.so -lm -shared -fno-omit-frame-pointer -fPIC -I/usr/include/python2.7/ -lpython2.7")
		command ="clang *.c -g3 -fsanitize=address -o ElsaABM -lm "
		command = "clang mABM.c mQuery.c mSector.c main.c mTest.c mUtility.c -g3 -fsanitize=address -o ElsaABM -lm "
	else:
		command = "gcc mABM.c main.c mQuery.c mSector.c mTest.c mUtility.c -o ElsaABM -lm"
		
	os.system(command)
	print "Executed this command:", command
