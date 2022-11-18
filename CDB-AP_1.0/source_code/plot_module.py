import matplotlib
matplotlib.use("TkAgg")  # backend of matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
""" This tab contains the basic outline for the plot found in most tabs.
Inherited from Ratio Curves, S and W parameter form, S vs. W plot, and S vs. W (Ref.) plot."""
LABEL_FONT_SIZE = 20
TICK_NUMBER_SIZE = 13
LINE_WIDTH = 2
LEGEND_FONT_SIZE = 14


class PlotWindow(tk.Frame):

    def __init__(self, name, data_container, *args, **kwargs):
        """ Parent class for the plots, since they all contain many of the same attributes """
        super().__init__(*args, **kwargs)
        self.columnconfigure(0, weight=1)

        self.name = name
        self.data_container = data_container  # needed for the checkbox states

        # example data - to be overwritten by children
        x = np.linspace(0, 10, 500)
        y = np.sin(x)
        self.data = {"Example Data":
                     np.array([x, y])}
        # default limits for the unfolded data
        self.ymin = 1e-6
        self.ymax = 1
        self.xmin = 460
        self.xmax = 560
        # to help the scale be correct for folding
        self.fold_value_changed = False

        # Define other variables for later
        self.inputs = {}  # a place to store all the input values - needed for limits
        self.canvas = ''
        self.ax = ''

        # refresh each page as you click on them
        self.bind("<FocusIn>", self.refresh)
        # When clicking on a tab without any data loaded, this code runs through the refresh functions several times,
        # causing a "No data" warning to pop up several times, including when navigating away from that tab. This
        # variable tracks whether the warning has been shown yet for a particular tab so that it only shows the warning
        # the first time through the refresh functions. This variable is reset to False for a tab when data is loaded
        # and the tab is selected.
        self.showed_no_data_warning = False

    def subframe1(self, ncols=None):
        """ Creates the first row with the refresh button, smoothing, and folding options.
         Not all of the windows will need this, but they won't hurt."""
        self.processing_buttons = tk.LabelFrame(self, text=None, background='gray93')
        # parent.rowconfigure(0, weight=1, uniform='row')
        if ncols:
            for n in range(ncols):
                self.processing_buttons.columnconfigure(n, weight=1)

        # the matplotlib plot is part of the first sub_frame
        self.canvas, self.ax = self.setup_plot_window()
        self.subframe1_buttons(self.processing_buttons)
        self.ax.tick_params(axis='x', direction="in")
        self.ax.tick_params(axis='y', direction="in")

        self.processing_buttons.grid(row=0, sticky="nsew")

    def subframe1_buttons(self, parent):
        refresh_button = ttk.Button(master=parent, text="Refresh", command=self.refresh)
        refresh_button.grid(row=0, column=0, sticky="nsew")

        self.inputs["smoothing_label"] = ttk.Label(parent, text="Smoothing Window Size")
        self.inputs["smoothing_label"].grid(row=0, column=1, sticky="nsew")

        self.inputs["Smoothing"] = ttk.Spinbox(parent, from_=1, to=1000, increment=1,
                                               textvariable=self.data_container.inputs["Smoothing"])
        # initialized in data_container
        self.inputs["Smoothing"].bind("<Return>", self.plot)
        self.inputs["Smoothing"].grid(row=0, column=2, sticky="nsew")

        self.inputs["GaussianSmoothing"] = ttk.Checkbutton(parent, text="Gaussian Smoothing",
                                                           variable=self.data_container.inputs[
                                                               "GaussianSmoothingState"])
        self.inputs["GaussianSmoothing"].grid(row=0, column=3, sticky="nsew")

        self.inputs["FoldButton"] = ttk.Checkbutton(parent, text="Fold",
                                                    variable=self.data_container.inputs["FoldingState"],
                                                    command=self.toggle_fold)
        self.inputs["FoldButton"].grid(row=0, column=4, sticky="nsew")

        self.inputs["ShiftButton"] = ttk.Checkbutton(parent, text="Shift Peaks",
                                                     variable=self.data_container.inputs["ShiftingState"])
        self.inputs["ShiftButton"].grid(row=0, column=5, sticky="nsew")

    def setup_plot_window(self):
        """ method 2 """
        plotFrame = tk.Frame(self)

        fig = Figure()  # figsize=(5, 6))
        ax = fig.add_axes([.1, .1, .8, .8])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # give legend room on the right
        ax.tick_params(direction='in', labelsize=TICK_NUMBER_SIZE)

        canvas = FigureCanvasTkAgg(fig, master=plotFrame)

        # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        self.toolbar = NavigationToolbar2Tk(canvas, plotFrame)
        self.toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        plotFrame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        return canvas, ax

    def plot(self):
        """ Adjusting font sizes can be done explicitly:
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        or all at once:
        plt.rcParams.update({'font.size': 22})
        """
        self.ax.clear()
        self.ax.plot(self.data["Example Data"][0],
                     self.data["Example Data"][1],
                     label="y=sin(x)")
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.ax.set_ylim(-1, self.ymax)  # ymin is hardcoded so that self.ymin can be something else
        self.canvas.draw()
        # update toolbar each time we redraw the figure to help it work more consistently
        self.toolbar.update()

    def subframe3(self, ncols=None):
        """ gray93 fills in the gaps to match the ttk widgets """
        parent = tk.LabelFrame(self, text=None, background='gray93')
        # parent.rowconfigure(0, weight=1, uniform='row')
        if ncols:
            for n in range(ncols):
                parent.columnconfigure(n, weight=1)

        # section for x and y axix
        ttk.Label(parent, text="y min").grid(row=1, column=0, sticky="nse")
        # no w term in sticky to get the labels to cling to the spinboxes
        self.inputs["ymin"] = ttk.Spinbox(parent, from_=-50, to=600, increment=0.1, width=10)
        self.inputs["ymin"].insert(tk.END, self.ymin)
        self.inputs["ymin"].bind("<Return>", self.plot)
        self.inputs["ymin"].grid(row=1, column=1, sticky="nsew")

        ttk.Label(parent, text="y max").grid(row=1, column=2, sticky="nse")
        self.inputs["ymax"] = ttk.Spinbox(parent, from_=-50, to=600, increment=0.1, width=10)
        self.inputs["ymax"].insert(tk.END, self.ymax)
        self.inputs["ymax"].bind("<Return>", self.plot)
        self.inputs["ymax"].grid(row=1, column=3, sticky="nsew")

        ttk.Label(parent, text="x min").grid(row=1, column=4, sticky="nse")
        self.inputs["xmin"] = ttk.Spinbox(parent, from_=-50, to=600, increment=0.1, width=10)
        self.inputs["xmin"].insert(tk.END, self.xmin)
        self.inputs["xmin"].bind("<Return>", self.plot)
        self.inputs["xmin"].grid(row=1, column=5, sticky="nsew")

        ttk.Label(parent, text="x max").grid(row=1, column=6, sticky="nse")
        self.inputs["xmax"] = ttk.Spinbox(parent, from_=-50, to=600, increment=0.1, width=10)
        self.inputs["xmax"].insert(tk.END, self.xmax)
        self.inputs["xmax"].bind("<Return>", self.plot)
        self.inputs["xmax"].grid(row=1, column=7, sticky="nsew")

        b = ttk.Button(parent, text="Use Limits from Plot", command=self.get_axis_limits)
        b.grid(row=1, column=8, sticky="nsew")

        # need to leave a row blank to get these last rows on the bottom
        parent.grid(row=3, sticky="nsew")

    def get_axis_limits(self):
        # get the x y limits from the plot
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        lims = (xmin, xmax, ymin, ymax)
        keys = ("xmin", "xmax", "ymin", "ymax")

        # place them into the input boxes
        for n in range(4):
            self.inputs[keys[n]].delete(0, tk.END)
            self.inputs[keys[n]].insert(tk.END, round(lims[n], 2))

    def refresh(self, *args):
        # include this as a way to change the data for the plot
        self.plot()

    def toggle_fold(self):
        self.fold_value_changed = True

    def create_reference_dropdown(self):
        """ used in ratiocurveplot and swref """
        subframe2 = tk.LabelFrame(self, text="Select a reference", background='gray93')
        # use only rowconfigure so the drop down box stays on the left
        subframe2.rowconfigure(0, weight=1)

        # self.reference = tk.StringVar()
        self.dropMenu = ttk.OptionMenu(subframe2, self.data_container.reference,
                                       "No Data", "No Data", "Please load some data")
        self.dropMenu.grid(row=0, column=0, sticky="e")

        subframe2.grid(row=2, column=0, sticky="nsew")

    def update_dropdown(self):
        """ used in ratiocurveplot and swref """
        # update the drop down menu
        self.dropMenu['menu'].delete(0, tk.END)
        choices = [self.data_container.get('label', sample=sample) for sample in self.data_container.get("keys")]
        if not choices:
            choices = ["No Data"]
            # only want to initialize this the first time
            self.data_container.reference.set(choices[0])
        else:
            if self.data_container.reference.get() == "No Data" or self.data_container.reference.get() == "None":
                self.data_container.reference.set(choices[0])
        for choice in choices:
            # TODO is there a more proper way to do this?
            # self.dropMenu['menu'].add_command(label=self.data_container.get('label', sample=choice),
            #                                   command=tk._setit(self.reference, choice))
            self.dropMenu['menu'].add_command(label=choice, command=tk._setit(self.data_container.reference, choice))


class LoadingScreen(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title('Loading')
        # self.geometry("260x50")
        self.update()

    def add_progress_bar(self):
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL,
                                            length=250, mode='determinate')
        self.progress_bar.pack(pady=20)
        self.update()

    def update_progress_bar(self, amount):
        self.progress_bar['value'] += amount
        self.update()


class Marker:

    def __init__(self, data_container):
        # choose from one of 24 markers
        self.title = "Markers"

        self.markers = {'s': 'square', 'o': 'circle', '^': 'triangle_up',
                        'v': 'triangle_down', 'D': 'diamond', '+': 'plus',
                        'x': 'x', '*': 'star', '<': 'triangle_left',
                        '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up',
                        'p': 'pentagon', 'h': 'hexagon1', 'H': 'hexagon2',
                        '8': 'octagon', '3': 'tri_left', '4': 'tri_right',
                        '.': 'point', 'P': 'plus(filled)', 'X': 'x(filled)',
                        'd': 'thin_diamond', '|': 'vline', '_': 'hline'}

        self.data_container = data_container

    def choose_marker(self, filename):
        # description = ['square', 'circle', 'triangle_up', 'triangle_down', 'diamond', 'plus', 'cross', 'star', ]
        self.filename = filename  # updates specific to data file.

        cell_width = 150  # 212
        cell_height = 22
        swatch_width = 48
        margin = 12
        topmargin = 40

        n = len(self.markers)
        self.ncols = 4
        self.nrows = n // self.ncols + int(n % self.ncols > 0)

        width = cell_width * 4 + 2 * margin
        height = cell_height * self.nrows + margin + topmargin
        dpi = 72

        self.fig, self.ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig.subplots_adjust(margin / width, margin / height,
                                 (width - margin) / width, (height - topmargin) / height)
        self.ax.set_xlim(0, cell_width * 4)
        self.ax.set_ylim(cell_height * (self.nrows - 0.5), -cell_height / 2.)
        self.ax.yaxis.set_visible(False)
        self.ax.xaxis.set_visible(False)
        self.ax.set_axis_off()
        self.ax.set_title(self.title, fontsize=24, loc="left", pad=10)

        for i, marker in enumerate(self.markers.keys()):
            row = i % self.nrows
            col = i // self.nrows
            y = row * cell_height

            swatch_start_x = cell_width * col
            text_pos_x = cell_width * col + swatch_width + 7

            self.ax.text(text_pos_x, y, self.markers[marker], fontsize=14,
                         horizontalalignment='left',
                         verticalalignment='center')
            self.ax.plot(swatch_start_x+swatch_width/2, y, marker, color='#1f77b4')

        # activate the buttons
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
        plt.show()

    # def on_click(self, event):
    def __call__(self, event):

        if event.inaxes != self.ax.axes:
            # safe guard against clicking outside the figure
            return

        x, y = event.xdata, event.ydata
        self.locate_marker(x, y)

    def locate_marker(self, x, y):
        xmin, xmax = self.ax.get_xlim()
        ymax, ymin = self.ax.get_ylim()
        xbins = np.linspace(xmin, xmax, self.ncols + 1)
        ybins = np.linspace(ymin, ymax, self.nrows + 1)

        # locate the bin
        xidx = np.where(xbins < x)[0][-1]
        yidx = np.where(ybins < y)[0][-1]

        codes = np.array(list(self.markers.keys())).reshape((self.ncols, self.nrows))
        # update the marker
        self.data_container.marker[self.filename].set(codes[xidx, yidx])

        self.fig.canvas.mpl_disconnect(self.cid)
        plt.close(self.fig)
