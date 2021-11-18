import tkinter as tk
from . import svswplot as s


class SWRef(s.SvsWPlot, tk.Frame):

    def __init__(self, name, data_container, *args, **kwargs):
        super().__init__(name, data_container, *args, **kwargs)

        self.label.destroy()
        self.label = tk.Label(self, text="Welcome to the S vs W plotting section with references")
        self.label.grid(row=0, column=0, padx=10, pady=10)
        self.name = name
        self.data_container = data_container

        self.subframe1()
        self.create_reference_dropdown()

        # specify the calculation for SW
        self.ref = True

    def refresh(self, *args):
        try:
            self.update_dropdown()
            # update the ratio curve data. The ratio itself is calculated later
            self.data = self.data_container.get("sw param data")
            # set the reference we selected
            self.data_container.set("reference",
                                    key=self.data_container.get('key', sample=self.data_container.reference.get()))

            super().refresh()
        except ValueError:
            tk.messagebox.showerror("Error", "Please load data first")

    def plot(self):
        super().plot()
