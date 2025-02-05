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
Joseph Watkins, 2020–2021,
Idaho National Laboratory

George Evans, 2021–Present,
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

Python Interpreter: 3.12 

Python Packages: tkinter, tkmacosx, Matplotlib, NumPy, Pandas, mplcursors

Compatible: Windows 10 and macOS

* Note:
   * CDB-AP 1.0 was built using python 3.8 and tested using python 3.10. This branch is being developed using the most up-to-date version of python. At time of writing, this branch is written using python 3.12.
   * CDB-AP 1.0 was tested extensively to be compatible with macOS. This branch has not yet been tested for compatibility with macOS, but it should be compatible regardless. 

General Notes
=============
This application was designed to accept MPA files that were produced using a Fast ComTech multiparameter analyzer and
the CSV files that were produced using TechnoAP hardware and software.
The application also accepts reduced data where the diagonal coincidence counts have been extracted by CDB-AP
or even by another program (if the data is formatted correctly).
Data imported by both means can be analyzed simultaneously and saved as an aggregate file. 

Written orginally using python 3.8. Updated using python 3.12. Running on macOS will require a third party download of python.
The Apple distributed version of Python contains bugs that may prevent it from working.
Issues have also been had with getting the Anaconda version to work with macOS.

Branch Update Notes
=============
An incomplete list of updates made to CDB-AP as part of this branch:
 * Replaced 'interp2d' (depreciated) with 'RectBivariateSpline' 
 * Rudementary ability to subtract TechnoAP CDB data sets from each other, allowing for calculating time-dependent data (In Progress; also needed for MPA files)
 * Added ability to export analyzed S Curves (normalized and not-normalized)
  * Includes adjustments for factors such as peak shifting and data smoothing
 * Added ability to calculate, display, and export statistical uncertainty on S curves and ratio curves
 * Better handling of bulk TechnoAP analysis (In Progress; also needed for MPA files)
 * Added Windows compatible executable for installing CDB-AP v1.0.0
  * macOS compatible application creatable if requested


Bugs
====
* Exported S and W parameters don't account for changes made, such as peak shift or data smoothing
* (In Progress)
