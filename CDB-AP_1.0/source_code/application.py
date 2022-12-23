""" Main driver for the application. Generate each tab"""

import tkinter as tk
from tkinter import ttk
from . import file_upload_form
from . import ratio_curves_plot
from . import SW_parameter_form
from . import S_vs_W_plot
from . import SvsW_ref_plot
from . import math_module as m


class Application(tk.Tk):
    """ Application root window. Controls all the actions and objectives. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("CDB Analysis Program")
        self.resizable(width=None, height=None)
        self.minsize(800, 600)

        ttk.Label(
            self,
            text="CDB Analysis Program",
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
        load_tab = file_upload_form.FileUploadForm(self.notebook, self.data_container)
        SW_parameter_tab = SW_parameter_form.SWParameterForm(self.notebook, self.data_container)
        ratio_curves_tab = ratio_curves_plot.RatioCurvesPlot(self.notebook, self.data_container)
        SvsW_plot_tab = S_vs_W_plot.SvsWPlot(self.notebook, self.data_container)
        SvsW_ref_tab = SvsW_ref_plot.SvsWRefPlot(self.notebook, self.data_container)

        self.notebook.add(load_tab, text="Load")
        self.notebook.add(SW_parameter_tab, text="S and W Parameters")
        self.notebook.add(ratio_curves_tab, text="Ratio Curves")
        self.notebook.add(SvsW_plot_tab, text="S vs. W")
        self.notebook.add(SvsW_ref_tab, text="S vs. W (Ref.)")
