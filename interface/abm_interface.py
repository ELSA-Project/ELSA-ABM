#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1,'../engine')
import os
from string import split

from ABMtactic import simulation



#inpt = CharArray(2);
main_dir = os.path.abspath('.')
main_dir = split(main_dir, '/src')[0]

inpt = ["", os.path.join(main_dir, "trajectories/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat"), os.path.join(main_dir, "results/output.dat")]


print "M1 source:", inpt[1]
print "Destination output:", inpt[2]
print
print
print "Running model..."
simulation(inpt)

print
print
print "Done."