#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest

from libs.genral_tools import *

# TODO: test find_entry_exit
class TrajConverterTest(unittest.TestCase):
	def setUp(self):
		self.Converter = TrajConverter()

	def test_conversion1(self):
		pass