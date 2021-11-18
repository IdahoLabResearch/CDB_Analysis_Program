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
            tk.messagebox.showerror("Error", "Please set/check SW parameters first")

    def plot(self):
        # we have to extract the data here because it can change any time we change the sw param tab
        SW = self.data_container.calculate_S(self.data, ref=self.ref)
        self.ax.clear()

        if not self.data_container.inputs["FlippingState"].get():
            xlabel = "W"
            ylabel = "S"
        else:
            xlabel = "S"
            ylabel = "W"

        s = [SW[key][xlabel] for key in SW]
        w = [SW[key][ylabel] for key in SW]
        # left click or hover to view, right click to hide.
        mplcursors.cursor(self.ax.plot(s, w, ','), hover=True)

        for key in SW:
            if self.data_container.hidden_state[key].get() == 'Visible':
                self.ax.plot(SW[key][xlabel], SW[key][ylabel], self.data_container.marker[key].get(),
                             label=self.data_container.get('label', key),
                             color=self.data_container.color[key].get())

        self.ax.set_ylabel(ylabel, fontsize=p.LABEL_FONT_SIZE)
        self.ax.set_xlabel(xlabel, fontsize=p.LABEL_FONT_SIZE)
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': p.LEGEND_FONT_SIZE})
        self.canvas.draw()
