import tkinter as tk
from tkinter import ttk
from . import PlotModule as p


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
        logscale_checkbox = ttk.Checkbutton(parent, text="Check to turn off log scale",
                                            variable=self.data_container.inputs["LogscaleState"])
        logscale_checkbox.grid(row=0, column=5, sticky='nsew')

    def create_sw_parameter_inputs(self):
        subframe2 = tk.LabelFrame(self, text="S-W Parameter Definitions", background='gray93')

        subframe2.rowconfigure(0, weight=1)  # single row frame
        for n in range(6):
            subframe2.columnconfigure(n, weight=1)

        ttk.Label(subframe2, text="S max").grid(row=0, column=0, sticky="nse")
        self.inputs["Smax"] = ttk.Spinbox(subframe2, from_=460, to=560, increment=0.01)
        self.inputs["Smax"].insert(tk.END, self.params["Smax"])
        self.inputs["Smax"].bind("<Return>", self.refresh)
        self.inputs["Smax"].grid(row=0, column=1, sticky="nsew")

        ttk.Label(subframe2, text="W min").grid(row=0, column=2, sticky="nse")
        self.inputs["Wmin"] = ttk.Spinbox(subframe2, from_=460, to=560, increment=0.01)  # cast to string before using
        self.inputs["Wmin"].insert(tk.END, self.params["Wmin"])
        self.inputs["Wmin"].bind("<Return>", self.refresh)
        self.inputs["Wmin"].grid(row=0, column=3, sticky="nsew")

        ttk.Label(subframe2, text="W max").grid(row=0, column=4, sticky="nse")
        self.inputs["Wmax"] = ttk.Spinbox(subframe2, from_=460, to=560, increment=0.01)
        self.inputs["Wmax"].insert(tk.END, self.params["Wmax"])
        self.inputs["Wmax"].bind("<Return>", self.refresh)
        self.inputs["Wmax"].grid(row=0, column=5, sticky="nsew")

        # calculate the companions in the data container
        subframe2.grid(row=2, sticky="nsew")

    def refresh(self, *args):
        try:
            # set new params
            for key in ("Smax", "Wmax", "Wmin"):
                self.data_container.set(name="parameter", key=key, value=float(self.inputs[key].get()))
                self.inputs[key].delete(0, tk.END)
                self.inputs[key].insert(tk.END, self.params[key])
            self.plot()
        except ValueError:
            tk.messagebox.showerror("Error", "Please load data first")

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

        # store the placeholder data
        self.data_container.set("placeholder data", data=df)

        # draw the lines for the sw parameters here
        [self.ax.vlines(val, self.ymin, self.ymax, linewidth=p.LINE_WIDTH) for val in self.params.values()]
        # alpha controls the transparency
        self.ax.axvspan(self.params["SmaxL"], self.params["Smax"], alpha=0.5, color='red')
        self.ax.axvspan(self.params["WmaxL"], self.params["WminL"], alpha=0.5, color='blue')
        self.ax.axvspan(self.params["Wmin"], self.params["Wmax"], alpha=0.5, color='blue')

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
