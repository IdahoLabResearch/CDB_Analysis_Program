import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
from .PlotModule import LoadingScreen, Marker
from tkinter import colorchooser
import os
import platform
if platform.system() == 'Darwin':
    from tkmacosx import Button as MacButton
    BUTTON_WIDTH = 110
    HIDDEN_WIDTH = 5
else:
    # account for Windows
    MacButton = tk.Button
    BUTTON_WIDTH = 10
    HIDDEN_WIDTH = 7


class FileUploadForm(tk.Frame):
    """ Here we have the user specify which files we are going to be using. """
    def __init__(self, parent, data_container, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # make the buttons scale with the size of the application
        # enable the check box spot to stretch
        self.rowconfigure(5, weight=1, uniform='row')

        # allow bottom row of buttons to stretch
        # for n in range(1, 5):
        #     self.columnconfigure(n, weight=1)
        self.columnconfigure((1, 2, 3, 4), weight=1)

        # Declare Variables
        self.new_file = True
        self.delete_tracker = {}
        self.del_button = {}  # included to help select them all later
        self.var = tk.IntVar
        self.loaded_file_frame = ''  # Placeholder

        self.inputs = {}  # to help track which save data to export
        # need track the changes made by the user to the columns/filenames.
        # Stores the Entry objects. Extract the new label with filename_labels[filename].get()

        # Need a structure to store the data
        self.data_container = data_container

        # list of matplotlib colors
        self.base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.current_base_color_idx = 0  # for default plotting

        # marker class for matplotlib
        self.markerclass = Marker(data_container)
        self.marker_list = list(self.markerclass.markers.keys())
        self.current_marker_idx = 0  # for default plotting
        self.image = {}

        fileinfo = tk.Label(self, text="CDB Analysis Program", background='gray93')
        fileinfo.config(font=("Courier", 44))
        fileinfo.grid(row=0, column=0, columnspan=5, sticky="nsew")

        # Create the buttons
        process_mpa_button = ttk.Button(self, text="Process a .mpa file", command=self.process_mpa)
        load_button = ttk.Button(self, text="Load Data", command=self.open_file)
        select_all = ttk.Button(self, text="Select all", command=self.select_all)
        delete_button = ttk.Button(self, text="Delete selected files", command=self.delete_files)
        # using a button to update the filenames because the focus out method was too confusing
        # to figure out in a reasonable amount of time
        rename_button = ttk.Button(self, text="Update filenames", command=self.update_names)
        # reset the labels in case the user makes a mistake
        reset_labels = ttk.Button(self, text="Reset filenames", command=self.reset_names)
        save_button = ttk.Button(self, text="Save raw data", command=self.save_file_data)

        # first column
        process_mpa_button.grid(row=1, column=0, sticky="nsew")
        load_button.grid(row=2, column=0, sticky="nsew")
        save_button.grid(row=4, column=0, sticky="nsew")
        self.save_options(row=5, column=0, rowspan=5)  # extend onto the bottom row

        # second column
        self.file_row = 1
        self.file_col = 1
        self.file_rowspan = 7  # to avoid resizing the buttons as more files are added
        self.file_columnspan = 4
        self.display_files()

        # bottom row
        select_all.grid(row=9, column=1, sticky="nsew")
        rename_button.grid(row=9, column=2, sticky="nsew")
        reset_labels.grid(row=9, column=3, sticky="nsew")
        delete_button.grid(row=9, column=4, sticky="nsew")

    def process_mpa(self):
        """ load a .mpa file and extract the 511 cdb peak """
        file_path = askopenfilename()
        self.update()  # to clear the dialog box
        # account for cancel button
        if file_path != "":
            # https://blog.furas.pl/python-tkinter-how-to-create-popup-window-or-messagebox-gb.html
            # https://www.youtube.com/watch?v=Grbx15jRjQA&ab_channel=Codemy.com

            loader = LoadingScreen(self)
            loader.add_progress_bar()

            # pass the loading bar into the first method because this is the longest process
            data2D, calibration = self.data_container.read_data2D(loader, file_path)
            x2i, y2i, data2i = self.data_container.reduce_data(data2D, calibration)
            combo = self.data_container.isolate_diagonal(x2i, y2i, data2i)
            loader.update_progress_bar(25)

            # extract the new file name
            # example of importing name 190128 - 10K0V+10K0F, 19J500.mpa
            # split from right to left, but only make one split
            # separator = '/'  # different in MacOS and Windows10
            # filename2_old = file_path.rsplit(separator, 1)[-1].rsplit(".", 1)[0] + "_511CDBpeak"
            # filepath_old = file_path.rsplit(separator, 1)[0]
            filename2 = os.path.split(file_path)[1].removesuffix(".mpa") + "_511CDBpeak"
            filepath = os.path.split(file_path)[0]
            # print("FILE_PATH", file_path)
            # print("FILENAME2:", filename2)
            # print("FILEPATH:", filepath)
            # print(filename2 == filename2_old)
            # print(filepath == filepath_old)

            # see if the user wants to change the name
            filename = asksaveasfilename(initialdir=filepath, initialfile=filename2, defaultextension="*.csv")
            loader.destroy()
            self.update()
            if filename != '':
                pd.DataFrame(combo).to_csv(filename, header=False, index=False)

                # with this successful, now we load the data into the application
                # this is the same sequence of commands from open_file
                new_df = pd.DataFrame(combo, columns=['x', filename])
                dict_data = self.data_container.from_df_to_dict(new_df)
                # store the data using the full file path
                self.data_container.set(name="raw data", key=filename, data=dict_data[filename])
                # save the short file name in filename_labels in the data_container
                self.data_container.set(name="update key", key=filename, new_key=filename2)
                self.display_files()

    def open_file(self):
        """ prompt the user to select a single file with data in it. """
        try:

            filename = askopenfilename()  # TODO add hover over instructions
            self.update()
            if filename != '':
                # need to check file type - could be space or comma separated
                header_present = self.check_for_header(filename)
                if not header_present:

                    with open(filename) as f:
                        line = f.readline()  # read one line
                        if ',' in line:
                            new_df = pd.read_csv(filename, header=None)
                        else:
                            new_df = pd.read_csv(filename, sep='   ', engine='python', header=None)

                    # load as pandas dataframes
                    new_df.columns = ["x", filename]
                    # combine and save as a dictionary
                    dict_data = self.data_container.from_df_to_dict(new_df)
                    # save the data as a dictionary for easy preservation
                    self.data_container.set(name="raw data", key=filename, data=dict_data[filename])
                    # don't forget to display!
                    self.display_files()

                else:
                    # header present indicates reloading old data.
                    # now read it back in
                    new_df = pd.read_csv(filename)
                    # convert to dictionary
                    d = self.data_container.from_df_to_dict(new_df)
                    # save to data structure
                    for key in d:
                        self.data_container.set(name="raw data", key=key, data=d[key])

                    # don't forget to display!
                    self.display_files()

        except ValueError:
            # todo replace with dialog box
            tk.messagebox.showerror("Error", "Open a single txt file. This button is not for reloading.")

    @ staticmethod
    def check_for_header(filename):
        """ open a file and check for a header. This should be sufficient for the files we will read """
        with open(filename) as f:
            # one line should suffice
            header = f.readline()  # passed in value is the number of total acceptable bytes.
            # identify if this is a header
            if ',' in header:
                # regular format for csv and data processed here
                header = header.split(',')
            else:
                # specific to the matlab saved data
                header = header.split('   ')[1:]  # extra space at the beginning of the line

            try:
                # now try casting each element to a float
                line = [float(el) for el in header]
                return False  # no header detected
            except ValueError:
                return True  # header detected

    def display_files(self):
        """
        Loops through the currently included files and displays them.
        Includes a delete button to remove data
         TODO replace the structure here with a listbox. This should be smoother and look
         better, but we need to keep the function to rename files, and access elements.
         """

        try:
            # to avoid pasting a frame on top of an old frame
            self.loaded_file_frame.destroy()
        except (NameError, AttributeError):
            # if there is no frame to destroy then we are fine.
            pass

        self.loaded_file_frame = tk.LabelFrame(self, text="Current loaded files", background='gray93')
        self.loaded_file_frame.rowconfigure(0, weight=1)
        files = self.data_container.get("keys")

        # need the max length
        # hard coded for now - just want this the sames size as the two buttons above
        # TODO find a way for this not to be hardcoded
        length = 80  # some filenames are really long and we don't want this bleeding off the page.

        # Stores the Entry objects. Extract the new label with filename_labels[filename].get()
        # we need this to be fresh every time. the needed info will be retrieved from the Math Module
        self.filename_labels = {}
        self.colors = {}
        self.hidden = {}  # to let the user choose if the data shows on the graph
        self.marker = {}  # store the marker buttons

        if len(files) == 0:
            # no data is loaded
            files = ["Ready to load some data!"]

        for n, filename in enumerate(files):
            self.delete_tracker[filename] = tk.IntVar()
            self.del_button[filename] = ttk.Checkbutton(self.loaded_file_frame, variable=self.delete_tracker[filename])
            self.del_button[filename].grid(row=n, column=0)
            # to get left aligned we anchor to the west side. However, this is only noticeable if
            # all of the boxes are the same size, hence width = 35
            # the file names tend to be around 26 - 30 characters long

            self.filename_labels[filename] = ttk.Entry(self.loaded_file_frame, width=length)
            self.filename_labels[filename].insert(tk.END, self.data_container.get('label', filename))
            self.filename_labels[filename].bind("<FocusOut>", self.update_names)
            self.filename_labels[filename].grid(row=n, column=1)

            # add a color option
            if filename == "Ready to load some data!":
                # print(filename)
                color = 'white'
                self.data_container.color[filename] = tk.StringVar(value="Choose Color")
            elif filename not in self.data_container.color.keys():
                # print(filename, "pos2")
                # assign a color if one is not assigned
                color = self.cycle_base_colors()  # only call once per button, as the color changes per call
                self.data_container.color[filename] = tk.StringVar(value=color)
            else:
                color = self.data_container.color[filename].get()

            # data=filename is to save the correct file with this button.
            # tkmacosx button width is in pixels
            # default colors for bg and fg cause errors.
            self.colors[filename] = MacButton(self.loaded_file_frame,
                                              command=lambda data=filename: self.change_data_color(data), width=BUTTON_WIDTH,
                                              bg=color, fg=color,
                                              highlightbackground=color,
                                              activebackground=color)
            self.colors[filename].grid(row=n, column=2)

            # add a marker option
            if filename not in self.data_container.marker.keys():
                marker = self.cycle_markers()
                self.data_container.marker[filename] = tk.StringVar(value=marker)  # defaults to circle

            self.marker[filename] = tk.Button(self.loaded_file_frame, textvariable=self.data_container.marker[filename],
                                              command=lambda data=filename: self.choose_marker(data),  # width=3,
                                              image=self.get_image_path(filename))
            self.marker[filename].grid(row=n, column=3)
            self.data_container.marker[filename].trace('w', lambda *args, c=filename: self.update_marker_image(c))

            # add a hidden options
            if filename not in self.data_container.hidden_state.keys():
                self.data_container.hidden_state[filename] = tk.StringVar()
                self.data_container.hidden_state[filename].set("Visible")
                current_state = "Visible"

            else:
                current_state = self.data_container.hidden_state[filename].get()

            self.hidden[filename] = ttk.OptionMenu(self.loaded_file_frame, self.data_container.hidden_state[filename],
                                                   current_state, "Visible", "Hidden")

            self.hidden[filename].config(width=HIDDEN_WIDTH)
            self.hidden[filename].grid(row=n, column=4)

        self.loaded_file_frame.grid(row=self.file_row, column=self.file_col,
                                    rowspan=self.file_rowspan, columnspan=self.file_columnspan, sticky="new")

    def change_data_color(self, filename):
        """ opens a color wheel to allow user to specify dataset color """
        color_code = colorchooser.askcolor(title="Choose color")
        color = color_code[1]
        # control the color of all parts of the button here.
        self.colors[filename].configure(bg=color, fg=color,
                                        highlightbackground=color,
                                        activebackground=color)

        if color_code[1] is not None:
            self.data_container.color[filename].set(color_code[1])

    def cycle_base_colors(self):
        desired_color = self.base_colors[self.current_base_color_idx]
        self.current_base_color_idx += 1

        if self.current_base_color_idx >= len(self.base_colors):
            # reset
            self.current_base_color_idx = 0
        return desired_color

    def cycle_markers(self):
        desired_marker = self.marker_list[self.current_marker_idx]
        self.current_marker_idx += 1

        if self.current_marker_idx >= len(self.marker_list):
            self.current_marker_idx = 0

        return desired_marker

    def choose_marker(self, filename):
        """ allows the user to select a marker for the sw plots """
        self.markerclass.choose_marker(filename)

    def get_image_path(self, filename):
        image = self.markerclass.markers[self.data_container.marker[filename].get()]
        # base_folder = os.path.dirname(__file__)  # need to get to sibling directory
        image_path = os.path.join("images", image + '.png')
        self.image[filename] = tk.PhotoImage(file=image_path)
        # saved as member variable to stop the image from being deleted by garbage collection
        return self.image[filename]

    def update_marker_image(self, filename, *args):
        # changes the image upon detection of a variable change
        self.marker[filename].configure(image=self.get_image_path(filename))

    def update_names(self, event=None):
        """ updates the list of short user defined labels. Can be called manually
        or by <FocusOut> on the specific label defined in display_files()"""
        # store the labels in the data container
        new_names = {key: val.get() for key, val in self.filename_labels.items()}
        self.data_container.set("update key", new_key=new_names)

    def reset_names(self):
        keys = self.data_container.get("keys")
        for key in keys:
            self.filename_labels[key].delete(0, tk.END)
            self.filename_labels[key].insert(tk.END, key)

        # now replace the labels in the data container.
        self.update_names()

    def select_all(self):
        selected = ('active', 'focus', 'selected', 'hover')  # not sure why this needs to be so long
        # loop through all the files that are currently loaded
        files = self.data_container.get("keys")
        for file in files:
            self.del_button[file].state(selected)
            self.delete_tracker[file].set(1)

    def save_options(self, row, column, rowspan=1, colspan=1):
        # save option box
        save_box = tk.LabelFrame(self, text="Processed data to save", background='gray93')
        save_box.rowconfigure(4, weight=1)  # need to configure this row with this weight to get it show on the bottom
        width = 20

        self.inputs["RatioCurveState"] = tk.IntVar()
        self.inputs["RatioCurves"] = ttk.Checkbutton(save_box, text="Ratio Curves",
                                                     variable=self.inputs["RatioCurveState"],
                                                     width=width)
        # ttk doesn't have the anchor parameter, but it is unneeded.
        self.inputs["RatioCurves"].grid(row=0, sticky='n')

        self.inputs["SWState"] = tk.IntVar()
        self.inputs["SW"] = ttk.Checkbutton(save_box, text="S and W", variable=self.inputs["SWState"],
                                            width=width)
        self.inputs["SW"].grid(row=1, sticky='n')

        self.inputs["SWRefState"] = tk.IntVar()
        self.inputs["SWRef"] = ttk.Checkbutton(save_box, text="S and W with reference",
                                               variable=self.inputs["SWRefState"], width=width)
        self.inputs["SWRef"].grid(row=2, sticky='n')

        self.inputs["ParamState"] = tk.IntVar()
        self.inputs["Params"] = ttk.Checkbutton(save_box, text="Parameters", variable=self.inputs["ParamState"],
                                                width=width)
        self.inputs["Params"].grid(row=3, sticky='n')

        # adding the save selected button
        export_button = ttk.Button(save_box, text="Save selected data", command=self.export_data)
        export_button.grid(row=4, column=0, sticky="sew")

        save_box.grid(row=row, column=column, rowspan=rowspan, columnspan=colspan, sticky="nsew")

    def export_data(self):
        """ If no data is loaded it will save an empty file """

        # for key in ("RatioCurveState", "SWState", "SWRefState", "ParamState"):
        #     state = self.inputs[key].get()

        if self.inputs["RatioCurveState"].get():
            filename = asksaveasfilename(initialdir=os.path.dirname(__file__), initialfile="RatioCurves", defaultextension="*.csv")
            self.update()
            if filename != '':
                data = self.data_container.get("ratio curves")
                # update the column names to reflect the user made changes
                old_columns = data.columns[1:]  # don't select the x column
                new_columns = [val.get() for val in self.filename_labels.values()]
                for n in range(len(new_columns)):
                    # this could be more efficient, but for now this works to change the column names
                    data = data.rename(columns={old_columns[n]: new_columns[n]})

                try:
                    data.to_csv(filename, index=False)
                except AttributeError:
                    tk.messagebox.showerror(title="Missing Data", message="Please load some data before saving.")

        if self.inputs["SWState"].get():
            filename = asksaveasfilename(initialdir=os.path.dirname(__file__), initialfile="S-W", defaultextension="*.csv")
            self.update()
            if filename != '':
                data = self.data_container.get("SW")
                # update the keys to reflect the user made changes
                data['sample'] = [val.get() for val in self.filename_labels.values()]
                data.set_index('sample', inplace=True)
                data.rename_axis(None, inplace=True)
                if len(data) != 0:
                    data.to_csv(filename)
                else:
                    tk.messagebox.showerror(title="Missing SW Data", message="Please load some data before saving.")

        if self.inputs["SWRefState"].get():
            filename = asksaveasfilename(initialdir=os.path.dirname(__file__), initialfile="S-W-Ref", defaultextension="*.csv")
            self.update()
            if filename != '':
                data = self.data_container.get("SW Ref")
                # update the keys to reflect the user made changes
                data['sample'] = [val.get() for val in self.filename_labels.values()]
                data.set_index('sample', inplace=True)
                data.rename_axis(None, inplace=True)

                if len(data) != 0:
                    data.to_csv(filename)
                else:
                    tk.messagebox.showerror(title="Missing SW Ref Data", message="Please load some data before saving.")

        if self.inputs["ParamState"].get():
            filename = asksaveasfilename(initialdir=os.path.dirname(__file__), initialfile="Parameters", defaultextension="*.csv")
            self.update()   # fixme supposed to keep window closed after user saves the file, but it doesn't work
            if filename != '':
                data = self.output_form()
                data.to_csv(filename)

    def save_file_data(self):
        """ Exports the dictionary containing the data to a single file """
        # TODO move to Applications class?
        # TODO work on this function after everything else works
        savefilename = asksaveasfilename(initialdir=os.path.dirname(__file__), initialfile="test", defaultextension="*.csv")
        self.update()
        if savefilename != '':
            data = self.data_container.get("raw data")
            df = self.data_container.from_dict_to_df(data)
            # temp solution - change headers to the short labels
            # fixme it appears that I have lost a few rows vs before I added these few lines
            # from here to the rename call.
            current_cols = df.columns
            d = {}
            for col in current_cols[1:]:
                d[col] = self.data_container.get('label', sample=col)

            df.rename(columns=d, inplace=True)

            df.to_csv(savefilename, index=False)

    def output_form(self):
        """ accesses and combines the various parameters and checkboxes and organizes
        them into one dataframe.
         Right now this function only works when we have refreshed every tab and
         clicked each button at least once (fold, shift, smoothing)
         """
        # check reference file
        ref = self.data_container.get("reference")

        if ref:
            output = pd.DataFrame()
            # loop through all the samples present
            samples = self.data_container.get("keys")
            for sample in samples:
                # each sample will get one row that stores all its unique data
                output = output.append(pd.DataFrame({'label': self.filename_labels[sample].get(),
                                                     # 'shift amount': self.data_container.get("shift amount", sample),
                                                     'color': self.data_container.color[sample].get(),
                                                     'marker': self.markerclass.markers[
                                                         self.data_container.marker[sample].get()],
                                                     'S': self.data_container.get("sw", sample)[0],
                                                     'W': self.data_container.get("sw", sample)[1],
                                                     'S Ref': self.data_container.get("sw ref", sample)[0],
                                                     'W Ref': self.data_container.get("sw ref", sample)[1],
                                                     }, index=[sample]))

            # add all the shared data (values will repeat for each row)
            output["Shift"] = self.data_container.get("is shifted")
            output["Folding"] = self.data_container.get("is folded")
            output["Smoothing Window Size"] = self.data_container.get("smoothing amount")
            ref_filepath = self.data_container.get("reference")  # we use the user label here.
            output["Reference Sample"] = self.data_container.get("label", sample=ref_filepath)

            # bounds for the sw parameters are shared
            parameters = self.data_container.get("parameters")
            for key in parameters:
                output[key] = parameters[key]

            return output

    def delete_files(self):
        keys = list(self.delete_tracker.keys())

        for key in keys:

            if self.delete_tracker[key].get():
                self.data_container.remove(key)
                self.delete_tracker.pop(key)
                self.filename_labels.pop(key)
                self.colors.pop(key)
                self.hidden.pop(key)

            if key == self.data_container.get("reference"):
                # need to change the reference in ratio curves
                data_keys = self.data_container.get('keys')

                # set the first data file to be the new reference
                if len(data_keys) != 0:
                    self.data_container.set("reference", key=data_keys[0])

        self.display_files()
