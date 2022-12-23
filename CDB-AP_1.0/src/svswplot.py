import tkinter as tk
from tkinter import ttk
from . import PlotModule as p
import mplcursors


class SvsWPlot(p.PlotWindow, tk.Frame):

    def __init__(self, name, data_container, *args, **kwargs):
        super().__init__(name, data_container, *args, **kwargs)

        self.rowconfigure(1, weight=1)  # let the plot resize

        self.label = tk.Label(self, text="Welcome to the S vs W plotting section")
        self.label.grid(row=0, column=0, padx=10, pady=10)
        self.name = name
        # load the data
        self.data_container = data_container

        # generate the top row of buttons
        self.subframe1()

        # specify the calculation for SW
        self.ref = False

    def subframe1_buttons(self, parent, *args):
        super().subframe1_buttons(parent, *args)

        # remove extra check-buttons since the data is generated in a different tab
        self.inputs["FoldButton"].destroy()
        self.inputs["ShiftButton"].destroy()
        self.inputs["GaussianSmoothing"].destroy()
        self.inputs["smoothing_label"].destroy()
        self.inputs["Smoothing"].destroy()

        # self.inputs["FlippingState"] = tk.IntVar()
        b3 = ttk.Checkbutton(parent, text="Flip Axis", variable=self.data_container.inputs["FlippingState"])
        b3.grid(row=0, column=1, sticky="nsew")

    def refresh(self, *args):
        try:
            self.data = self.data_container.get("sw param data")
            super().refresh()  # just a call to self.plot
        except KeyError:
            # Check if this is the first time that this warning appears to patch the problem that the program
            # goes through this function multiple times when the messagebox is used, causing the error message
            # to pop up multiple times each click
            if not self.showed_no_data_warning:
                self.showed_no_data_warning = True
                tk.messagebox.showerror("Error", "Please set/check SW parameters first")

    def plot(self):
        # we have to extract the data here because it can change any time we change the sw param tab
        SW, SW_err = self.data_container.calculate_S(self.data, ref=self.ref)  # . added sw)err
        self.ax.clear()

        if not self.data_container.inputs["FlippingState"].get():
            xlabel = "W"
            ylabel = "S"
        else:
            xlabel = "S"
            ylabel = "W"

        # s = [SW[key][xlabel] for key in SW]  # . todo delete commented out code
        # w = [SW[key][ylabel] for key in SW]
        # hover_labels = ["S = " + str(SW[key][xlabel]) + " +/- " + str(SW_err[key]["dS"]) +
        #                 "\nW = " + str(SW[key][ylabel]) + " +/- " + str(SW_err[key]["dW"]) for key in SW]
        # print(hover_labels, "hv")
        # left click or hover to view, right click to hide.
        # cursor = mplcursors.cursor(self.ax.plot(s, w, ','), hover=True, multiple=True)  # . added multiple = true
        # cursor.connect("add", lambda sel: sel.annotation.set_text(hover_labels[sel.index()]))

        for key in SW:
            if self.data_container.hidden_state[key].get() == 'Visible':
                # self.ax.plot(SW[key][xlabel], SW[key][ylabel], self.data_container.marker[key].get(),
                #              label=self.data_container.get('label', key), # . todo delete commented out code
                #              color=self.data_container.color[key].get())
                self.ax.errorbar(SW[key][xlabel], SW[key][ylabel], fmt=self.data_container.marker[key].get(),
                             yerr=SW_err[key]["dS"], xerr=SW_err[key]["dW"], label=self.data_container.get('label', key),
                             color=self.data_container.color[key].get())  # . added this error bar stuff
        mplcursors.cursor(self.ax, hover=True, multiple=True)  # . added here commented out above

        self.ax.set_ylabel(ylabel, fontsize=p.LABEL_FONT_SIZE)
        self.ax.set_xlabel(xlabel, fontsize=p.LABEL_FONT_SIZE)
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': p.LEGEND_FONT_SIZE})
        self.canvas.draw()

        self.showed_no_data_warning = False  # Reset the error warning for no data once data has been loaded
