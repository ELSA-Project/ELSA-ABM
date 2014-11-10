#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1,'../engine')
from ElsaABM import iter_sim

print "Running model"
iter_sim(("../traffic_generation/M1/inputABM_n-10_Eff-0.975743921611_Nf-1500.dat"), ("output.dat"))
print "Done."