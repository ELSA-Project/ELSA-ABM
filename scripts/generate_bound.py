#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

with open('../libs/All_shapes_334.pic','r') as f:
	all_shapes = pickle.load(f)
	boundary = list(all_shapes['LFMM']['boundary'][0].exterior.coords)
	assert boundary[0]==boundary[-1]

with open('../abm_tactical/config/bound_latlon.dat', 'w') as f:
	for x, y in boundary:
		f.write(str(x) + '\t' + str(y) + '\n')