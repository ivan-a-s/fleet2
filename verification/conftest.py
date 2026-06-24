"""
Configure pytest to run from the fleet2 root directory.

model.py loads 'vehicle_modelling/surrogates.json' via a relative path at import
time, so the working directory must be the fleet2 root when pytest collects tests.
This file is executed before any test module is imported.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
