#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the model. You can safely run this several times if you do not mess up too much
with libs/paths.py.
"""

import os
from os.path import join as jn

if __name__ == '__main__':
	main_dir = os.path.split(os.path.abspath(__file__))[0]

	print "Please type the path of the main directory for results. The path can be relative."
	print "It will be created if it does not exist."
	ans = raw_input("Leave blank (press enter) to use the default, which is ../results\n")

	print "Writing libs/paths.py file..."
	if ans=='':
		ans = '../results'
		
	result_dir = os.path.abspath(ans)

	with open(os.path.join(main_dir, 'libs/paths.py')) as f:
		lines = f.readlines()

	new_lines = []
	found = False
	for l in lines:
		if 'result_dir' in l and l[0]!='#':
			new_lines.append('result_dir = "' + result_dir + '"')
			found = True
		else:
			new_lines.append(l)

	if not found:
		new_lines.append('')
		new_lines.append('result_dir = "' + result_dir + '"')

	with open(jn(main_dir, 'libs', 'paths.py'), 'w') as f:
		for line in new_lines:
			f.write(line)

	os.system('mkdir -p ' + result_dir)

	os.system('mkdir -p ' + jn(result_dir, 'networks'))
	os.system('mkdir -p ' + jn(result_dir, 'trajectories'))
	os.system('mkdir -p ' + jn(result_dir, 'trajectories', 'M1'))
	os.system('mkdir -p ' + jn(result_dir, 'trajectories', 'M3'))

	os.system('cp ' + jn('abm_strategic', 'paras.py') + ' '+ jn('abm_strategic', 'my_paras.py'))
	os.system('cp ' + jn('abm_strategic', 'paras_iter.py') + ' '+ jn('abm_strategic', 'my_paras_iter.py'))
	
	print
	print "Compiling C code and making wrapper..."
	os.system("cd abm_tactical && ./compile.py")

	print
	print "Writing Documentation..."
	os.system("./make_doc.py")


