============================
 CDB Analysis Program
============================

Description
===========

This program provides a convenient tool to load and analyze data
from coincidence Doppler broadening (CDB) positron annihilation spectroscopy (PAS) data.
Find more details about CDB-AP v1.0.0 in the SoftwareX publication 
`“CDB-AP: An application for coincidence Doppler broadening spectroscopy analysis.” <https://doi.org/10.1016/j.softx.2023.101475>`_

Features
--------
* Provides a means to rapidly reduce coincidence Doppler broadening data.
* Rapidly compares and analyzes many datasets simultaneously. 
* Generates an S-W plot for several samples from a file.
* Creates ratio curves.
* Calculates S and W parameters based on user input or optimization.

Authors
=======
Joseph Watkins, 2020–2021
Idaho National Laboratory

George Evans, 2021–2022
Idaho National Laboratory

Installation
============
1.	Select the button “Code” near the top of this directory.
2.	Select “Download ZIP” from the dropdown menu that appears.
3.	Extract the downloaded file to a folder of your choice.
4.	Open the command line, then move to the directory “CDB-AP_1.0” within the downloaded folder.
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
This application was designed to accept MPA files that were produced using a Fast ComTech multiparameter analyzer and
the CSV files that were produced using TechnoAP hardware and software.
The application also accepts reduced data where the diagonal coincidence counts have been extracted by CDB-AP
or even by another program (if the data is formatted correctly).
Data imported by both means can be analyzed simultaneously and saved as an aggregate file. 

Written using python 3.8. Running on macOS will require a third party download of python.
The Apple distributed version of Python contains bugs that may prevent it from working.
Issues have also been had with getting the Anaconda version to work with macOS.

Bugs
====
* Exported S and W parameters don't account for changes made, such as peak shift or data smoothing

  * The S and W parameters in the S and W parameter graph do account for user input

* 'interp2d' function has been removed in SciPy 1.14.0

  * To avoid this bug, do one of the following:

    * Use an earlier version of SciPy 

    * Download python files for CDB-AP from the GitHub branch "CDB_Analysis_Program_with_Time_Dependence"

    * Download and install the executable "Install CDB-AP v1.0.0 (Windows).exe" from the GitHub branch "CDB_Analysis_Program_with_Time_Dependence"

* (In progress)
