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

George Evans, 2021-2022
Idaho National Laboratory

Installation
============
1.	Select the button “Code” near the top of this directory.
2.	Select “Download ZIP” from the dropdown menu that appears.
3.	Extract the downloaded file to a folder of your choice.
4.	Open the command line, then move to the directory “HyPAT” within the downloaded folder.
5.	Run the following code to open the application: python main.py

 
* Note:
   * Your system may require the following code instead: python3 main.py
   * These instructions assume you already have Python and the required Python packages installed.
   * An executable that will install the application is available upon request (no python required)

Python Interpreter: 3.8, 3.9, or 3.10

Python Packages: tkinter, tkmacosx, Matplotlib, NumPy, Pandas, mplcursors

Compatible: macOS and Windows 10

General Notes
=============
This application was designed to accept .mpa files that were produced using a Fast ComTech multi parameter analyzer and the .csv files that were produced using TechnoAP hardware and software. The application also accepts reduced data where the diagonal coincidence counts have been extracted (by another program). Data imported by both means can be analyzed simultaneously and saved as an aggregate file. 

Written using python 3.8. Running on macOS will require a third party download of python.
The Apple distributed version of Python contains bugs that may prevent it from working. Issues have also been had with getting the Anaconda version to work with macOS.

Bugs
====
The code sections are not yet populated.
