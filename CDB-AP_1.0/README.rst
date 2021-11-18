============================
 CDB Analysis Program
============================

Description
===========

This program provides a convenient tool to load and analyze data
from coincidence doppler broadening positron annihilation spectroscopy data.

Features
--------
* Provides a means to rapidly reduce coincidence Doppler broadening data.
* Rapidly compare and analyze many datasets simultaneously. 
* Generate an S-W plot for several samples from a file.
* Creates ratio curves.
* Calculates S and W parameters based on user input or optimization.

Authors
=======
Joseph Watkins, 2020-2021
Idaho National Laboratory

Requirements
============

* Python 3.8
* Tkinter
* tkmacosx
* Matplotlib
* Numpy
* Pandas

Installation
============
Include installation information for matplotlib, pandas, and numpy here
I will find out as we go if we for sure need numpy or not.

Usage
=====

To start the application, run::

  python3 PAS_CDB_GUI/pas_cdb_gui.py

General Notes
=============
This application was designed to accept .mpa files that were produced using a Fast ComTech   multi parameter analyzer. This native data file is a large two-dimensional matrix of values, from which the coincidence counts can be extracted (where detector 1 + detector 2 = 1022 keV). The application also accepts reduced data where the diagonal coincidence counts have been extracted (by another program). Data imported by both means can be analyzed simultaneously and saved as an aggregate file. 

Written using python 3.8. Running on macOS will require a third party download of python.
The Apple distributed version of Python contains bugs that may prevent it from working. Issues have also been had with getting the Anaconda version to work with macOS.

Bugs
====
The code sections are not yet populated.