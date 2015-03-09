# -*- coding: utf-8 -*-
"""
Paths for utilities or libraries.
"""
import os
from os.path import join, split, dirname

path_ksp = '/home/earendil/Programmes/YenKSP-master'
path_codes = '/home/earendil/Documents/ELSA/Codes/'
path_utilities = '/home/earendil/Documents/ELSA/Utilities/'
path_modules = join(split(dirname(__file__))[0], 'libs')

main_dir = os.path.abspath(__file__)
main_dir = os.path.split(os.path.dirname(main_dir))[0]

result_dir = os.path.join(os.path.dirname(main_dir), 'results')
