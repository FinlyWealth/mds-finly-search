"""
Root conftest.py to handle path setup for all tests.
"""
import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root) 