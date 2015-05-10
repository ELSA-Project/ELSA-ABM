# -*- coding: utf-8 -*-
"""
Paths for utilities or libraries. DO NOT MODIFY.
"""

import sys
sys.path.insert(1, '..')
from os.path import join, dirname, abspath, split

main_dir = abspath(__file__)
main_dir = split(dirname(main_dir))[0]

# Do not modify this
path_codes = join(main_dir, 'libs')
path_utilities = join(main_dir, 'libs')
path_modules = join(main_dir, 'libs')

# Overwrite this to indicate your main directory for results here
result_dir = join(dirname(main_dir), 'results')