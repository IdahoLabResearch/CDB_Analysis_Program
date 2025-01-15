import tkinter as tk
from tkinter import ttk
from . import plot_module as p


class SWParameterForm(p.PlotWindow): #, tk.Frame):

    def __init__(self, name, data_container, *args, **kwargs):
        """ Two label frames one for refresh & toggle smoothing
        One for the parameter adjusting"""
        super().__init__(name, data_container, *args, **kwargs)

        # make the buttons scale with the size of the application
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # line one
        self.name = name
        self.data_container = data_container

        # load the data, and a place to hold the parameters
        self.data = self.data_container.get("raw data")
        self.inputs = {}
        # self.smoothing = False
        # self.logscale = True

        # line two
        self.subframe1(ncols=6)

        # line three
        # collect defining parameters
        self.params = self.data_container.get("parameters")
        self.create_sw_parameter_inputs()

        # line four
        self.subframe3(ncols=8)

    def subframe1_buttons(self, parent):
        super().subframe1_buttons(parent)

        self.inputs["FoldButton"].destroy()  # this causes the blank white space where the row element would be
        self.inputs["ShiftButton"].grid(row=0, column=4, sticky="nsew")  # slide this button over

        # new buttons
        logscale_checkbox = ttk.Checkbutton(parent, text="Linear Scale",
                                            variable=self.data_container.inputs["LogscaleState"])
        logscale_checkbox.grid(row=0, column=5, sticky='nsew')

    def create_sw_parameter_inputs(self):
        subframe2 = tk.LabelFrame(self, text="S and W parameter regions of interest (ROI) limits (keV)",
                                  background='gray93')

        subframe2.rowconfigure(0, weight=1)  # single row frame
        for n in range(6):
            subframe2.columnconfigure(n, weight=1)

        ttk.Label(subframe2, text="S-ROI max:").grid(row=0, column=0, sticky="nse")
        self.inputs["S-ROI Max (keV)"] = ttk.Spinbox(subframe2, from_=461, to=561, increment=0.01)
        self.inputs["S-ROI Max (keV)"].insert(tk.END, self.params["S-ROI Max (keV)"])
        self.inputs["S-ROI Max (keV)"].bind("<Return>", self.refresh)
        self.inputs["S-ROI Max (keV)"].grid(row=0, column=1, sticky="nsew")

        ttk.Label(subframe2, text="Right W-ROI min:").grid(row=0, column=2, sticky="nse")
        self.inputs["Right W-ROI Min (keV)"] = ttk.Spinbox(subframe2, from_=461, to=561, increment=0.01)  # cast to string before using
        self.inputs["Right W-ROI Min (keV)"].insert(tk.END, self.params["Right W-ROI Min (keV)"])
        self.inputs["Right W-ROI Min (keV)"].bind("<Return>", self.refresh)
        self.inputs["Right W-ROI Min (keV)"].grid(row=0, column=3, sticky="nsew")

        ttk.Label(subframe2, text="Right W-ROI max:").grid(row=0, column=4, sticky="nse")
        self.inputs["Right W-ROI Max (keV)"] = ttk.Spinbox(subframe2, from_=461, to=561, increment=0.01)
        self.inputs["Right W-ROI Max (keV)"].insert(tk.END, self.params["Right W-ROI Max (keV)"])
        self.inputs["Right W-ROI Max (keV)"].bind("<Return>", self.refresh)
        self.inputs["Right W-ROI Max (keV)"].grid(row=0, column=5, sticky="nsew")

        # calculate the companions in the data container
        subframe2.grid(row=2, sticky="nsew")

    def refresh(self, *args):
        try:
            # set new params
            for key in ("S-ROI Max (keV)", "Right W-ROI Max (keV)", "Right W-ROI Min (keV)"):
                self.data_container.set(name="parameter", key=key, value=float(self.inputs[key].get()))
                self.inputs[key].delete(0, tk.END)
                self.inputs[key].insert(tk.END, self.params[key])
            self.plot()
        except ValueError:
            # Check if this is the first time that this warning appears to patch the problem that the program
            # goes through this function multiple times when the messagebox is used, causing the error message
            # to pop up multiple times each click
            if not self.showed_no_data_warning:
                self.showed_no_data_warning = True
                tk.messagebox.showerror("Error", "Please load data first")

        # store the parameters that were used for this instance.
        self.data_container.check_boxes["fold"] = self.data_container.inputs["FoldingState"].get()
        self.data_container.check_boxes["shift"] = self.data_container.inputs["ShiftingState"].get()
        self.data_container.check_boxes["smoothing_window_size"] = self.data_container.inputs["Smoothing"].get()
        self.data_container.check_boxes["gaussian_smoothing"] = self.data_container.inputs["GaussianSmoothingState"].get()

    def plot(self, *args):
        """ event occurs when Return is pressed, but only is passed in some of the times this function will be called"""
        self.ax.clear()

        # don't add the keys that are hidden
        keys = [key for key in self.data if self.data_container.hidden_state[key].get() == 'Visible']

        # step one: convert to dataframe (will be done sooner later)
        df = self.data_container.from_dict_to_df(self.data)

        for key in keys:
            if self.data_container.inputs["GaussianSmoothingState"].get():
                df[key] = df[key].rolling(window=int(self.data_container.inputs["Smoothing"].get()), center=True,
                                          min_periods=1, win_type='gaussian').mean(std=1)

            else:
                df[key] = df[key].rolling(window=int(self.data_container.inputs["Smoothing"].get()), center=True,
                                          min_periods=1).mean()

        if self.data_container.inputs["ShiftingState"].get():
            df = self.data_container.shift_data_to_match_peaks(df, folding=False)

        # make the plot here
        for key in keys:
            # allow for default colors as well as user defined
            # if self.data_container.color[key].get() == "Choose Color":
            #     color = None
            # else:
            #     color = self.data_container.color[key].get()
            self.ax.plot(df['x'], df[key], label=self.data_container.get('label', key), linewidth=p.LINE_WIDTH,
                         color=self.data_container.color[key].get())

        # store the S curve data # . used to be called "placeholder data"
        self.data_container.set("s curves", data=df)

        # draw the lines for the sw parameters here
        [self.ax.vlines(val, self.ymin, self.ymax, linewidth=p.LINE_WIDTH) for val in self.params.values()]
        # alpha controls the transparency
        self.ax.axvspan(self.params["S-ROI Min (keV)"], self.params["S-ROI Max (keV)"], alpha=0.5, color='red')
        self.ax.axvspan(self.params["Left W-ROI Max (keV)"], self.params["Left W-ROI Min (keV)"], alpha=0.5, color='blue')
        self.ax.axvspan(self.params["Right W-ROI Min (keV)"], self.params["Right W-ROI Max (keV)"], alpha=0.5, color='blue')

        # set the legend so it is outside the main box
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': p.LEGEND_FONT_SIZE})
        if self.data_container.inputs["LogscaleState"].get():
            self.ax.set_yscale('linear')
        else:
            self.ax.set_yscale('log')
            self.ax.tick_params(axis='y', direction="in")
        self.ax.set_ylim(float(self.inputs["ymin"].get()), float(self.inputs["ymax"].get()))
        self.ax.set_xlim(float(self.inputs["xmin"].get()), float(self.inputs["xmax"].get()))
        self.ax.set_xlabel("Energy (keV)", fontsize=p.LABEL_FONT_SIZE)
        self.ax.set_ylabel("Counts", fontsize=p.LABEL_FONT_SIZE)
        self.canvas.draw()
        self.showed_no_data_warning = False  # Reset the error warning for no data once data has been loaded
