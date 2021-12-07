""" Main driver for the application. Generate each tab"""

import tkinter as tk
from tkinter import ttk
from . import fileuploadform
from . import ratiocurveplot
from . import swparameterform
from . import svswplot
from . import swref
from . import MathModule as m


class Application(tk.Tk):
    """ Application root window. Controls all the actions and objectives. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("CDB Plotting Application")
        self.resizable(width=None, height=None)
        self.minsize(800, 600)

        ttk.Label(
            self,
            text="CDB Plotting Application",
            font=("TkDefaultFont", 16)
        ).grid(row=0)

        self.data_container = m.DoTheMathStoreTheData()
        self.notebook = ttk.Notebook()
        self.load_pages()
        self.notebook.grid(row=0, sticky='nsew')

        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)  # make row zero of top level window stretchable
        top.columnconfigure(0, weight=1)  # make column zero of top level window stretchable

    def load_pages(self):
        """ store the initial code to load each tab """
        load_tab = fileuploadform.FileUploadForm(self.notebook, self.data_container)
        SW_info_tab = swparameterform.SWParameterForm(self.notebook, self.data_container)
        ratio_curve_tab = ratiocurveplot.RatioCurvePlot(self.notebook, self.data_container)
        SvsW_plot_tab = svswplot.SvsWPlot(self.notebook, self.data_container)
        SWRef_tab = swref.SWRef(self.notebook, self.data_container)

        self.notebook.add(load_tab, text="Load")
        self.notebook.add(SW_info_tab, text="SW Params")
        self.notebook.add(ratio_curve_tab, text="Ratio Curves")
        self.notebook.add(SvsW_plot_tab, text="S vs W")
        self.notebook.add(SWRef_tab, text="S/W Ref")
