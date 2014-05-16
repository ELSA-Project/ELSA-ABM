#! /usr/bin/env python
# -*- coding: utf-8 -*-

from shutil import copy

list_of_external_files = ['/home/earendil/Documents/ELSA/Modules/general_tools.py', 
							'/home/earendil/Documents/ELSA/Distance/tools_airports.py']

for f in list_of_external_files:
	copy(f , '.')