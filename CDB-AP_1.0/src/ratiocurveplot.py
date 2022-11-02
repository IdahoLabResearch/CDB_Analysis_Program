import tkinter as tk
from tkinter import ttk
import numpy as np
from . import PlotModule as p


class RatioCurvePlot(p.PlotWindow, tk.Frame):

    def __init__(self, name, data_container, *args, **kwargs):
        super().__init__(name, data_container, *args, **kwargs)

        self.rowconfigure(1, weight=1)  # enable the plot window to change size as we scale

        self.name = name
        self.data_container = data_container
        self.choices = []
        self.inputs = {}
        self.fold = True
        self.fold_value_changed = True
        self.xmin = -0.1
        self.xmax = 50
        self.ymin = 0
        self.ymax = 1.4

        self.subframe1(ncols=7)  # calls subframe1buttons

        self.create_reference_dropdown()
        self.subframe3(ncols=8)  # options for adjusting x and y limits
        # create a second axis
        self.ax2 = self.ax.twiny()
        self.ax2.format_coord = self.make_format(self.ax2, self.ax)  # allow for both coordinates to be visible
        self.ax2.tick_params(direction='in', labelsize=p.TICK_NUMBER_SIZE)

    def subframe1_buttons(self, parent):
        super().subframe1_buttons(parent)

        # add a button to include or remove the straight line for the reference file.
        # Default is to include it
        self.inputs["ReferenceLine"] = ttk.Checkbutton(parent, text="Remove Reference Line",
                                                       variable=self.data_container.inputs["ReferenceLineState"])
        self.inputs["ReferenceLine"].grid(row=0, column=6, sticky='nsew')

    def refresh(self, *args):
        try:
            self.update_dropdown()
            super().refresh()
            # self.showed_no_data_warning = False
        except ValueError:
            # Check if this is the first time that this warning appears to patch the problem that the program
            # goes through this function multiple times when the messagebox is used, causing the error message
            # to pop up multiple times each click
            if not self.showed_no_data_warning:
                self.showed_no_data_warning = True
                print("VALUE ERROR")  # / todo deleted
                tk.messagebox.showerror("Error", "Please load data first")

        # store the parameters that were used for this instance.
        # the only data this affects is the ratio curves so we only need to save it here
        self.data_container.check_boxes["fold"] = self.data_container.inputs["FoldingState"].get()
        self.data_container.check_boxes["shift"] = self.data_container.inputs["ShiftingState"].get()
        self.data_container.check_boxes["smoothing_window_size"] = self.data_container.inputs["Smoothing"].get()

    @staticmethod
    def make_format(current, other):
        """ adapted from https://stackoverflow.com/questions/21583965/matplotlib-cursor-value-with-two-axes
        This conversion should work for both twinx and twiny.
        I only changed the Left and Right labels to Energy and Momentum, respectively"""
        def to_au(x):
            # for the conversion to the new scale.
            # this axis is purely visual
            return x * 3.92 / 7.28

        # current and other are axes
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.transData.transform((x, y))  # gets the numbers to show
            inv = other.transData.inverted()
            # convert back to data coords with respect to ax
            ax_coord = inv.transform(display_coord)
            coords = [ax_coord, (to_au(x), y)]
            # < forces left alight
            # 40 indicates size of space after variable
            return ('Energy: {:<40}    Momentum: {:<}'
                    .format(*['({:.3f}, {:.3f})'.format(x, y) for x, y in coords]))

        return format_coord

    def plot(self, *args):
        self.ax.clear()
        try:
            # only create it in this function
            self.ax2.clear()
        except AttributeError:
            pass

        ref = self.data_container.get('key', sample=str(self.data_container.reference.get()))  # gets the key for the reference
        df = self.data_container.calc_ratio_curves(ref, window_size=int(self.data_container.inputs["Smoothing"].get()),
                                                   folding=self.data_container.inputs["FoldingState"].get(),
                                                   shift=self.data_container.inputs["ShiftingState"].get(),
                                                   drop_ref=self.data_container.inputs["ReferenceLineState"].get(),
                                                   gauss=self.data_container.inputs["GaussianSmoothingState"].get())

        # plot the data
        for col in df.columns[1:]:
            if self.data_container.hidden_state[col].get() == 'Visible':
                # allow for default colors as well as user defined
                # if self.data_container.color[col].get() == "Choose Color":
                #     color = None
                # else:
                #     color = self.data_container.color[col].get()
                self.ax.plot(df['x'], df[col], label=self.data_container.get('label', col), linewidth=p.LINE_WIDTH,
                             color=self.data_container.color[col].get())
        if self.fold_value_changed:
            names = ('ymin', 'ymax', 'xmin', 'xmax')
            if self.data_container.inputs["FoldingState"].get():
                values = (0, 1.4, -0.1, 50)
            else:
                values = (0, 2, 460, 560)
            for n in range(4):
                self.inputs[names[n]].delete(0, tk.END)
                self.inputs[names[n]].insert(tk.END, values[n])
            self.fold_value_changed = False

        # create a second x axis - just the new ticks
        def to_au(x):
            new_vals = x * 3.92 / 7.28
            return ["{:.3f}".format(x) for x in new_vals]

        self.ax2.set_xlabel("momentum (a.u.)")
        self.ax2.set_xlim(float(self.inputs["xmin"].get()), float(self.inputs["xmax"].get()))
        # set the tick locations
        new_ticks = np.linspace(float(self.inputs["xmin"].get()), float(self.inputs["xmax"].get()), 5)

        self.ax2.set_xticks(new_ticks)
        self.ax2.set_xticklabels(to_au(new_ticks))  # ax2 is purely visual

        self.ax.set_ylim(float(self.inputs["ymin"].get()), float(self.inputs["ymax"].get()))
        self.ax.set_xlim(float(self.inputs["xmin"].get()), float(self.inputs["xmax"].get()))
        self.ax.set_xlabel("Energy", fontsize=p.LABEL_FONT_SIZE)
        self.ax.set_ylabel("Counts", fontsize=p.LABEL_FONT_SIZE)
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': p.LEGEND_FONT_SIZE})
        self.canvas.draw()
        self.showed_no_data_warning = False  # Reset the error warning for no data once data has been loaded
