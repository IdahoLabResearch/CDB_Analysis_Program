"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['pas_cdb_gui.py']
DATA_FILES = [('', ['images'])]
OPTIONS = {'packages': ['pandas', 'numpy', 'scipy', 'tkmacosx', 'matplotlib', 'mplcursors']}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
