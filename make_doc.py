#!/usr/bin/env python

import os

os.system("pydoc -w abm_strategic")
os.system("mv abm_strategic.html doc/")

reps = ['abm_strategic', 'abm_tactical', 'interface']

for rep in reps:
	os.system("cd " + rep)
	for root, dirs, files in os.walk(rep):
		for f in files:
			if f[-3:] == '.py':
				name = f[:-3]#rep + "." + f[:-3]
				os.system("cd " + rep + " && pydoc -w " + name)
				os.system("mv " + rep + '/' + name + '.html doc/' + rep + '.' + name + '.html')
				#os.system("mv " + name + ".html" + " doc/" + name + ".html")
	
