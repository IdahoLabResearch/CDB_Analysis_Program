import numpy as np
import pandas as pd
import re
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, interp2d
from tkinter.messagebox import askokcancel
import tkinter as tk


# noinspection PyUnresolvedReferences
class DoTheMathStoreTheData:
    """ serves as a container for the data and controls all the calculations """
    def __init__(self):
        """
        conversion from matlab defaults
        params = np.array([0.382, 1.0, 4.0])
        params /= 3.92/7.28
        params += 511
        left_params = 511 - abs(params-511)
        """

        self.data = {}  # Store the raw data
        self.C_norms = {}  # Store the normalizing constants  # . added this
        self.filename_labels = {}  # placeholder for the short labels for each data set
        self.ratio_curves = ''  # current form of the ratio curves will be stored here as a pandas dataframe
        self.sw_param_data = pd.DataFrame()
        self.SW = pd.DataFrame()
        self.SW_err = pd.DataFrame()  # . added this
        self.SWRef = pd.DataFrame()
        self.check_boxes = pd.DataFrame({"fold": False,
                                         "shift": False,
                                         "smoothing_window_size": 1,
                                         "gaussian_smoothing": False}, index=["CheckButtons"])
        self.shift_values = {}
        self.hidden_state = {}  # user may choose to hide data
        self.color = {}  # store the color for each data set
        self.marker = {}  # datapoint shape for SW plots

        # store the state of the smoothing parameter check buttons
        self.inputs = {"ReferenceLineState": tk.IntVar(),
                       "LogscaleState": tk.IntVar(),
                       "GaussianSmoothingState": tk.IntVar(),
                       "FoldingState": tk.IntVar(),
                       "ShiftingState": tk.IntVar(),
                       "Smoothing": tk.StringVar(value=1),
                       "FlippingState": tk.IntVar()}

        # reference file used in the ratio curve tab and the sw ref tab
        self.reference = tk.StringVar()  # contains the full file path

        # set the parameters on the right
        self.parameters = {"S-ROI Max (keV)": 511.77, "Right W-ROI Min (keV)": 513.02, "Right W-ROI Max (keV)": 519.08}
        # calculate the parameters on the left
        key = ("S-ROI Max (keV)", "Right W-ROI Min (keV)", "Right W-ROI Max (keV)")
        new_key = ("S-ROI Min (keV)", "Left W-ROI Min (keV)", "Left W-ROI Max (keV)")
        for n in range(3):
            # use the displacement from center to maintain symmetry
            self.parameters[new_key[n]] = 511 - abs(511 - self.parameters[key[n]])

    def get(self, name="raw data", sample=None):
        """ Returns specific data """
        name = name.lower()  # just to be safe (sends the string to all lowercase)
        if name == "raw data":
            return self.data

        elif name == "sw param data":
            return self.sw_param_data

        elif name == "key":
            if sample in self.filename_labels.values():
                # loop through dictionary to find the value then return key
                for filepath, label in self.filename_labels.items():
                    if sample == label:
                        return filepath
            elif sample in self.data.keys():
                # handle the case where the user has not shortened the name
                return sample
            # else:
                # tk.messagebox.showerror('Error', 'label not found when displaying the reference filename')

        elif name == "keys":
            return [key for key in self.data.keys()]

        elif name == "ratio curves":
            return self.ratio_curves

        elif name == "parameters":
            return self.parameters

        elif name == "sw":
            if sample:
                # return self.SW.loc[sample, "S"], self.SW.loc[sample, "W"]
                if sample in self.SW.index:
                    return self.SW.loc[sample, "S"], self.SW.loc[sample, "W"]  # returns a tuple
                else:
                    return 0, 0
            else:
                return self.SW

        elif name == "sw ref":
            if sample:
                # return self.SvsWRefPlot.loc[sample, "S"], self.SvsWRefPlot.loc[sample, "W"]
                if sample in self.SWRef.index:
                    return self.SWRef.loc[sample, "S"], self.SWRef.loc[sample, "W"]
                else:
                    return 0, 0
            else:
                return self.SWRef

        elif name == "check box values":
            return self.check_boxes

        elif name == "is shifted":
            val = self.check_boxes.loc["CheckButtons", "shift"]
            if val == 0:
                return False
            elif val == 1:
                return True
            else:
                return val

        elif name == "shift amount":
            # needs a filename passed in
            N = len(self.get("keys"))
            if len(self.shift_values) == N:
                return self.shift_values[sample]
            else:
                return 0

        elif name == "is folded":
            val = self.check_boxes.loc["CheckButtons", "fold"]
            if val == 0:
                return False
            elif val == 1:
                return True
            else:
                return val

        elif name == "smoothing amount":
            return self.check_boxes.loc["CheckButtons", "smoothing_window_size"]

        elif name == "gaussian smoothing":
            return self.check_boxes.loc["CheckButtons", "gaussian_smoothing"]

        elif name == "reference":
            return self.reference.get()

        elif name == "label":
            if sample in self.filename_labels.keys():
                return self.filename_labels[sample]
            elif sample in self.data.keys():
                # user has not declared a nickname yet
                self.set("update key", key=sample, new_key=sample)
                return sample
            elif sample in ["Ready to load some data!", "No Data"]:
                return sample
            else:
                tk.messagebox.showerror('Error', "column name not in filename labels")

        elif name == "c_norm":  # . added this (although not sure if it is needed)
            return self.C_norms

        else:
            tk.messagebox.showerror('Error', "No variable by the name {}.".format(name))

    def set(self, name, key=None, data=None, value=None, new_key=None, c_norm=None):  # . added c_norm stuff
        name = name.lower()
        if name == "raw data":
            # TODO check the data type to ensure compatibility
            if key is None or data is None:
                tk.messagebox.showerror('Error', "Missing information in set function: "+str(name))
            self.data[key] = data
        elif name == "parameter":
            if key is None or value is None:
                tk.messagebox.showerror('Error', "Missing information in set function: "+str(name))

            # TODO add checks to make the value is allowed
            # we want to keep the parameters symmetrical.
            # if the user enters a value for the right side, detect it and adjust
            xpeak = 511
            distance_to_peak = abs(value - xpeak)  # to set the symmetry value
            # loop through three cases
            left_options = ("S-ROI Min (keV)", "Left W-ROI Min (keV)", "Left W-ROI Max (keV)")
            for n, option in enumerate(("S-ROI Max (keV)", "Right W-ROI Min (keV)", "Right W-ROI Max (keV)")):
                if key == option:
                    # check size
                    if value < xpeak:
                        # supposed to be greater
                        self.parameters[left_options[n]] = round(value, 2)
                        self.parameters[option] = round(xpeak + distance_to_peak, 2)
                    else:
                        self.parameters[option] = round(value, 2)
                        self.parameters[left_options[n]] = round(xpeak - distance_to_peak, 2)
                elif key not in ("S-ROI Max (keV)", "Right W-ROI Min (keV)", "Right W-ROI Max (keV)"):
                    tk.messagebox.showerror('Error', "Invalid key")
        elif name == "smoothing":
            if value is None:
                tk.messagebox.showerror('Error', "No value passed in")
            else:
                self.check_boxes["smoothing_window_size"] = value

        elif name == "placeholder data":
            # this is where we store the data created in the sw params tab
            # TODO update the efficiency
            # df = self.from_dict_to_df(data)  # not ready to use a dataframe yet
            self.sw_param_data = data

        elif name == "reference":
            self.reference.set(key)
            # tk.messagebox.showerror('Error', "Set", key, "as the new reference file.")

        elif name == "update key":
            # want to use this function for two things
            # updating all at once or a single file at a time
            if isinstance(new_key, dict):
                # overwrites the old set
                self.filename_labels = new_key

            elif isinstance(new_key, str):
                # updates a single value
                self.filename_labels[key] = new_key
            else:
                tk.messagebox.showerror('Error', "Not a dictionary or a string: "+str(type(new_key)))

        elif name == "c_norm":  # . added this
            if key is None or c_norm is None:
                tk.messagebox.showerror('Error', "Missing information in set function: "+str(name))
            self.C_norms[key] = c_norm
            print("C_norm is", self.C_norms[key], "for", key)

    def remove(self, key):
        """ Needed to remove extra data at the request of the user. """
        # remove the data
        self.data.pop(key)
        # remove the short label
        self.filename_labels.pop(key)
        # remove hidden and color variables
        self.hidden_state.pop(key)
        self.color.pop(key)
        self.marker.pop(key)

    def calculate_S(self, df, ref=False):
        """ Currently accepts a dictionary containing the data
        ref toggles the location the sw data is saved. """
        SW_idx = {}
        SW = {}
        SW_err = {}  # . added
        print("norms", self.C_norms)  # . todo del

        # collect the indices for the SW bounds - they all share the same x axis
        for key in ("Left W-ROI Max (keV)", "S-ROI Min (keV)", "Right W-ROI Min (keV)"):
            SW_idx[key] = np.where(df['x'] <= self.parameters[key])[0][-1]
        for key in ("Left W-ROI Min (keV)", "S-ROI Max (keV)", "Right W-ROI Max (keV)"):
            SW_idx[key] = np.where(df['x'] >= self.parameters[key])[0][0]

        if ref:
            # calculate the ref first
            Sref = (np.trapz(df[self.reference.get()][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"] + 1],
                             df['x'][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"] + 1]) /
                    np.trapz(df[self.reference.get()], df['x'])
                    )
            Wref = ((np.trapz(df[self.reference.get()][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"] + 1],
                              df['x'][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"] + 1]) +
                     np.trapz(df[self.reference.get()][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"] + 1],
                              df['x'][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"] + 1])) /
                    np.trapz(df[self.reference.get()], df['x'])
                    )

        for col in df.columns[1:]:  # . added unc and num stuff to this for loop
            print()
            # calculate S and W for that data set
            S = (np.trapz(df[col][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"]+1],
                          df['x'][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"]+1]) /
                 np.trapz(df[col], df['x']))

            W = ((np.trapz(df[col][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"] + 1],
                           df['x'][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"] + 1]) +
                  np.trapz(df[col][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"] + 1],
                           df['x'][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"] + 1])) /
                 np.trapz(df[col], df['x']))

            # N_S = sum(df[col][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"] + 1]) * self.C_norms[col]
            # N_W = (sum(df[col][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"]+1]) +
            #        sum(df[col][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"]+1])) * self.C_norms[col]
            # N_total = sum(df[col]) * self.C_norms[col]
            N_W = ((np.trapz(df[col][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"] + 1],
                             df['x'][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"] + 1]) +
                    np.trapz(df[col][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"] + 1],
                             df['x'][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"] + 1]))) \
                  * self.C_norms[col]
            N_S = (np.trapz(df[col][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"] + 1],
                            df['x'][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"] + 1])) * self.C_norms[col]
            N_total = np.trapz(df[col], df['x']) * self.C_norms[col]
            print("N_total =", N_total)
            print("N_S/N_total", N_S/N_total, "vs. calculated S", S)
            print("N_W/N_total", N_W/N_total, "vs. calculated W", W)

            dS = (N_S / N_total) * np.sqrt(1 / N_S + 1 / N_total)
            dW = (N_W / N_total) * np.sqrt(1 / N_W + 1 / N_total)
            print("ds:", dS, "& dW:", dW)
            print("np.trapz(df[col], df['x']) =", np.trapz(df[col], df['x']))

            # print("S", S, np.trapz(df[col][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"]+1],
            #       df['x'][SW_idx["S-ROI Min (keV)"]:SW_idx["S-ROI Max (keV)"]+1]), np.trapz(df[col], df['x']),
            #       "W", W, (np.trapz(df[col][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"]+1],
            #       df['x'][SW_idx["Left W-ROI Max (keV)"]:SW_idx["Left W-ROI Min (keV)"]+1]) + np.trapz(df[col][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"]+1],
            #       df['x'][SW_idx["Right W-ROI Min (keV)"]:SW_idx["Right W-ROI Max (keV)"]+1])), np.trapz(df[col], df['x']))  # . TODO Delete print here and above

            if ref:
                S /= Sref
                W /= Wref

            SW[col] = {"S": S, "W": W}
            SW_err[col] = {"dS": dS, "dW": dW}  # . added

            # don't want to duplicate any entries
            if not ref:
                rows = [val for val in self.SW.index]
            else:
                rows = [val for val in self.SWRef.index]

            if col in rows:
                # overwrite the old values
                if not ref:
                    self.SW.loc[col, "S"] = S
                    self.SW.loc[col, "W"] = W
                else:
                    self.SWRef.loc[col, "S"] = S
                    self.SWRef.loc[col, "W"] = W
                self.SW_err.loc[col, "dS"] = dS  # . added this
                self.SW_err.loc[col, "dW"] = dW  # . added this
            else:
                # safe to add to the data frame
                if not ref:
                    self.SW = self.SW.append(pd.DataFrame({"S": S, "W": W}, index=[col]))
                else:
                    self.SWRef = self.SWRef.append(pd.DataFrame({"S": S, "W": W}, index=[col]))
                self.SW_err = self.SW_err.append(pd.DataFrame({"dS": dS, "dW": dW}, index=[col]))  # . added this
        return SW, SW_err  # . added sw_err

    def calc_ratio_curves(self, ref_key, window_size=1, folding=True, shift=True, drop_ref=False, gauss=False):
        """
        Driver method for calculating the ratio curves
        contains function calls to the major steps of the process.
        # TODO optimize std
        """
        # self.reference = ref_key
        # start with a df and stick to it.
        df = self.from_dict_to_df(self.data)

        # smoothing
        df = self.smooth_the_data2(df, window_size, gauss=gauss)
        self.set("smoothing", value=window_size)  # store the value

        # we have a couple routes we can take
        if shift:
            df = self.shift_data_to_match_peaks(df, folding=folding)
        elif not shift and folding:
            # this function still uses a dictionary
            data_to_fold = self.from_df_to_dict(df)
            folded_data = {}
            for key, data in data_to_fold.items():
                folded_data[key] = self.fold_the_data(data)
            # send back to df
            df = self.from_dict_to_df(folded_data)

        # calculate the ratio curves
        df = self.apply_reference(df, ref_key, drop_ref=drop_ref)

        # store the result
        self.ratio_curves = df

        return df

    def shift_data_to_match_peaks(self, df, folding=False):
        """ pass in a data frame that has gone through lineupdata()"""
        # we first need to separate the df into several dfs, since each x axis will shift a different amount
        dfs = {}
        for col in df.columns[1:]:
            dfs[col] = df.loc[:, ['x', col]]  # isolate the two columns of interest
            # now that we have each sample in its own dataframe, we can apply the shift
            shift = self.get_shift(dfs[col])
            dfs[col]['x'] += shift

            # save the shift value
            self.shift_values[col] = shift  # note this is a dictionary

        # now simplify the data back down to a single dataframe that shares the x axis
        df_interp = self.lineup_shifted_data(dfs, folding=folding)

        return df_interp

    @staticmethod
    def smooth_the_data2(df, window_size, gauss=False, std=1):
        # smooth the y columns, but not x
        for col in df.columns[1:]:

            if not gauss:
                df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()
            else:
                df[col] = df[col].rolling(window=window_size, center=True, min_periods=1,
                                          win_type="gaussian").mean(std=std)
            # std=1, , win_type='gaussian'
        return df

    @staticmethod
    def fold_the_data(data, units="Energy"):
        """ accepts data in a numpy array [[#,#],[#,#],[#,#],...,[#,#]]
        This function is for one sample at a time
        We also pass in the amount the data has been shifted"""
        # TODO I want to replace this method with the one I wrote in DataFoldingRework
        # just need to verify that the data is overall about the same
        combo = np.array([data[:, 0] - 511, data[:, 1]]).transpose()  # shift back to difference frame
        # the actual zero point only shows up if we haven't shifted any of the data

        try:
            zp, = np.where(combo[:, 0] == 0)[0]  # find the peak
        except ValueError:
            # Just in case exact equality isn't found. (It should be for data that uses this function)
            zp = np.argmin(abs(combo[:, 0] - 0))
        halfwidth = np.min([zp, len(combo) - zp + 1])  # find the middle
        avcombo = np.zeros([halfwidth, 2])  # container for the averages
        avcombo[:, 0] = abs(combo[zp-1:zp + halfwidth, 0])  # keep the right half of the data
        # this line is trying to say end and beginning, but ensures that each array ends up the same number of elements
        # add one to the reversed array since python doesn't grab the last index
        avcombo[:, 1] = (combo[zp-1:zp + halfwidth, 1] + combo[zp - halfwidth + 1:zp + 1, 1][::-1]) / 2
        # normalize the data
        avcombo[:, 1] = avcombo[:, 1] / np.trapz(avcombo[:, 1], avcombo[:, 0])

        if units == "Momentum":
            # this section converts to momentum units.
            p = avcombo[:, 0] * 3.92 / 7.28
            n = avcombo[:, 1] / np.trapz(avcombo[:, 1], p)  # normalize with the momentum units

            return np.array([p, n])
        else:
            # Energy units
            # no need to normalize again
            # new_df = pd.DataFrame(np.array([p, n]).transpose())
            return avcombo

    @staticmethod
    def apply_reference(df, reference, drop_ref=False):
        """ reference must be filename
        same but just with dataframe"""
        df2 = df.div(df[reference], axis='index')
        if drop_ref:
            df2.drop(reference, axis=1, inplace=True)
        df2[df2.columns[0]] = df[df.columns[0]]
        return df2

    @staticmethod
    def get_shift(df):
        # need amplitude, center, and standard deviation
        peak_idx = np.argmax(df[df.columns[1]])
        center = df[df.columns[0]][peak_idx]
        amplitude = df[df.columns[1]][peak_idx]
        std_dv = np.std(df[df.columns[0]])

        def gaussian(x, amp, cen, sigma):
            return amp * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen) / sigma) ** 2)))

        popt, pcov = curve_fit(gaussian, df[df.columns[0]], df[df.columns[1]], p0=[amplitude, center, std_dv])

        # now that this works, let's shift the xaxis
        # need distance from peak to 511
        shift = 511 - popt[1]
        # all we need to do now is add this shift to the x axis
        return shift

    @staticmethod
    def lineup_shifted_data(dfs, x='x', units="Energy", folding=False):
        """ Accepts a list (dictionary) of pandas dataframes,
         returns a single dataframe, with one x axis. """
        # we don't want to extrapolate due to the noisy data, so we will select the inner most x min/max
        xmins = []
        xmaxes = []
        lengths = []  # simplest course of action will be to take the average length
        for df in dfs.values():
            xmin = min(df[x])
            xmax = max(df[x])
            xmins.append(xmin)
            xmaxes.append(xmax)
            lengths.append(len(df[df.columns[0]]))
        # select largest min and smallest max
        xmin = max(xmins)
        xmax = min(xmaxes)
        # We won't keep the original spacing, but we'll try to maintain the original number of points.
        # We want one over lapping point at 511
        # We will create two arrays
        # in this example there are 1325 data points - should be pretty consistent for the matlab data
        sides = (xmin, xmax)
        smaller_side = np.argmin([511-xmin, xmax-511])
        difference = abs(511 - sides[smaller_side])
        num_points = int(np.ceil(np.mean(lengths)/2))  # if the length is odd we will round up
        # the duplicated value at 511 will preserve the old peak counts
        left_x = np.linspace(511-difference, 511, num_points)
        right_x = np.linspace(511, 511+difference, num_points)

        # last bit is to interpolate to get the x axes to match
        if folding:
            df_interp = pd.DataFrame({x: right_x})
        else:
            df_interp = pd.DataFrame({x: np.concatenate([left_x[:-1], right_x])})

        for key, df in dfs.items():  # dfs is a dictionary of dataframes
            # create the interpolation function
            f = interp1d(df[x], df[key], kind='linear')  # linear should give us closest to the same values
            # use this function to populate the y axis for the left and the right x axes
            left_y = f(left_x)
            right_y = f(right_x)
            
            if folding:
                flipped_left_y = left_y[::-1]  # folding occurs here
                mean_y = (flipped_left_y + right_y) / 2
                # the last step is to normalize the data
                df_interp[key] = mean_y / np.trapz(mean_y, right_x)
            else:
                df_interp[key] = np.concatenate([left_y[:-1], right_y])

        if units == "Momentum":
            # adjust the x axis and renormalize
            df_interp[x] -= 511
            df_interp[x] *= 3.92 / 7.28
            df_interp[key] = df_interp[key] / np.trapz(df_interp[key], df_interp[x])
        elif folding:
            df_interp[x] -= 511

        return df_interp

    @staticmethod
    def lineupdata(dfs, X, debug=False, eps=1e-3):
        """ expects a list of pandas dataframes
            X is a string containing the name of the x axis"""

        first_vals = np.array([d[X].iloc[0] for d in dfs])
        if debug:
            tk.messagebox.showerror('Error', "Top row: "+str(first_vals))

        # main algorithm is here
        # need a short list of unique elements in the first column
        first_vals_unique = np.unique(first_vals)
        if debug:
            tk.messagebox.showerror('Error', "Unique elements: "+str(first_vals_unique))

        # identify which columns contain which values
        indices = {}
        for item in first_vals_unique:
            indices[item], = np.where(abs(first_vals - item) < 1e-3)
            if debug:
                tk.messagebox.showerror('Error', "Indices of {}: {}".format(item, indices[item]))

        min_idx = np.argmin(first_vals)  # to identify which column has the smallest starting value
        # duplicates don't matter - this should just take the first occurrence
        if debug:
            tk.messagebox.showerror('Error', "Min: "+str(min_idx))  # dfs[min_idx]
        min_df = dfs[min_idx]

        # use this min to identify how much to shift the other columns
        # if the first value matches the min we don't need to shift - results in shift_amount=0
        shift_amount = []
        for value in first_vals_unique:
            # find the index of this value in the min dataframe
            # can use pandas index since there is only one value to find
            # todo make sure the tolerance is set to a good value. I think 10^-3 is enough.
            idx = min_df[X].index[abs(min_df[X] - value) < eps].to_list()
            shift_amount.append(idx[0])
            if debug:
                tk.messagebox.showerror('Error', "shift: "+str(idx[0]))

        # now apply the shift amounts to the correct dataframes
        # we have the first vals stores as the key to a dictionary
        for n, first_row_value in enumerate(indices):
            if debug:
                tk.messagebox.showerror('Error', "N, val: {}, {}".format(n, first_row_value))
            # each key links us to a list
            for df_idx in indices[first_row_value]:
                dfs[df_idx] = dfs[df_idx].shift(shift_amount[n])
                if debug:
                    tk.messagebox.showerror('Error', "DF: "+str(df_idx))

        dflong = pd.concat([d for d in dfs], axis=1)
        dflong.dropna(inplace=True)

        # from here we need to only keep one x column, but the rest of the y columns
        # ensure that all the x columns have the same name, but the y columns are unique
        df_final = dflong.loc[:, ~dflong.columns.duplicated()]

        return df_final

    @staticmethod
    def from_df_to_dict(df):
        """ accepts a pandas dataframe (x, y1, y2...) and converts to
        a dictionary with numpy arrays"""
        cols = df.columns
        x = cols[0]
        keys = cols[1:]

        data = {}
        for key in keys:
            data[key] = np.array([
                            df[x].to_numpy(),
                            df[key].to_numpy()
            ]).transpose()

        return data

    def from_dict_to_df(self, d):
        """ accepts a python dictionary
        converts it to a list of pandas dataframes, then sends through lineupdata

        The dictionary must be of the shape
            {key: [[x1, y1],
                   [x2, y2],
                   ...
                   [xn, yn]],
            key2: [[...]]}

        Works for transposed numpy arrays
        """
        dfs = []
        for key in d:
            df = pd.DataFrame({"x": d[key][:, 0],
                               key: d[key][:, 1]})
            dfs.append(df)
        dflong = self.lineupdata(dfs, "x")

        return dflong

    @staticmethod
    def read_data2D(loader, file_path):
        """

        :param loader:
            updated after the file is loaded into the program
            after looping through the file to identify sections?
            then after the 2d section is sliced out
            then after it is converted to an array
        :param file_path:
        :return:
        """
        with open(file_path) as f:
            lines = f.readlines()
        loader.update_progress_bar(25)
        # read in the header information
        ADC1, ADC2, callines = 0, 0, 4
        cal = np.zeros([2, 4])  # calibration information
        # collects header length, det1, det2, and 2d data lengths
        for n, line in enumerate(lines):
            # just read all of them; we'll end before too long
            if '[ADC1]' in line:
                ADC1 = 1  # on line 45

            # now save the calibration for 1
            if ADC1 == 1 and 'caluse=1' in line:
                for i in range(callines):
                    cal[0][i] = float(lines[n + 1 + i][8:])
                ADC1 = 0  # reset since we won't record this again

            # continuing to read down can identify 2
            if '[ADC2]' in line:
                ADC2 = 1

            # now save the calibration for 2
            if ADC2 == 1 and 'caluse=1' in line:
                for i in range(callines):
                    cal[1][i] = float(lines[n + 1 + i][8:])
                ADC2 = 0  # reset

            # save the length of the header section
            if '[DATA0' in line:
                header_len = n
                # get length of detector 1 data
                det1length = int(re.sub('[^0-9]', '', line[7:]))  # removes non-numeric characters

            if '[DATA1' in line:
                # get length of detector 2 data
                det2length = int(re.sub('[^0-9]', '', line[7:]))

            if '[CDAT0' in line:
                # get length of 2D data
                det2Dlength = int(re.sub('[^0-9]', '', line[7:]))
                break
        # save 2d data to an array for later use
        data2D = lines[header_len + 1 + det1length + 1 + det2length + 1:]  # 4.26 seconds
        loader.update_progress_bar(25)
        # np.fromiter(a, dtype=float) is about 3 seconds faster than np.array(a, dtype=float)
        # however it only works for one dimensional arrays.
        data2D = np.fromiter(data2D, dtype=float)  # 5.43 seconds
        loader.update_progress_bar(25)
        # convert to 2d array
        data2D = data2D.reshape((det1length, det2length)).transpose()  # reshape is done a little differently
        data2D[0, 0] = 0  # eliminate the spike

        return data2D, cal

    @staticmethod
    def reduce_data(data2D, cal, interp_step):
        # default to read the calibration saved in mpa
        # this produces the coefficients for a 1st order polynomial  (linear)
        det1cal = np.polyfit(cal[0][::2], cal[0][1::2], 1)
        det2cal = np.polyfit(cal[1][::2], cal[1][1::2], 1)

        # create a 2d grid for plotting
        x, y = np.meshgrid(np.arange(1, len(data2D[0]) + 1),
                           np.arange(1, len(data2D[:, 0]) + 1))  # include the endpoints

        # scale x and y according to the linear fit
        y = det1cal[1] + det1cal[0] * y  # I guess it is arbitrary which detector goes to which variable
        x = det2cal[1] + det2cal[0] * x

        # this data set is too large to use practically, and we only care about the center anyway
        # isolate the square bound by 461 < x, y < 561
        lower_bound = 461
        upper_bound = 561

        # find the interval such that 461 < x,y < 561
        lx = np.where(x[0] > lower_bound)[0][0]  # isolate the leftmost element
        rx = np.where(x[0] < upper_bound)[0][-1]  # isolate the rightmost element
        ly = np.where(y[:, 0] > lower_bound)[0][0]
        ry = np.where(y[:, 0] < upper_bound)[0][-1]

        # isolate the square of interest - note python drops the last index
        rx += 1
        ry += 1
        x2 = x[ly:ry, lx:rx]
        y2 = y[ly:ry, lx:rx]
        data2D = data2D[ly:ry, lx:rx]

        x2i, y2i = np.meshgrid(np.arange(max(x2[0][0], y2[0][0]), min(x2[-1][-1], y2[-1][-1]), interp_step),
                               np.arange(max(x2[0][0], y2[0][0]), min(x2[-1][-1], y2[-1][-1]), interp_step))
        f = interp2d(x2[0], y2[:, 0], data2D)  # defaults to linear interpolation
        data2i = f(x2i[0], y2i[:, 0])

        return x2i, y2i, data2i

    @staticmethod
    def isolate_diagonal(x2i, y2i, data2i, interp_step):
        # the next step is to isolate that diagonal
        max_loc = np.argmax(data2i)  # flattens the array and returns the index of the max value
        r = max_loc // len(data2i)  # integer division returns the row index
        c = max_loc % len(data2i)  # modulus gives the remainder from the division, giving the column
        Epeak = x2i[r, c] + y2i[r, c]  # sum of coords is the energy of the photon that hit the detector
        epsilon = 2  # width of the box is 2 keV
        E1E2 = x2i + y2i  # sum of the energy at each location
        # pd.DataFrame(E1E2).to_excel("E1E2 pre.xlsx")  # . todo delete this
        E1E2 = E1E2.flatten()
        # pd.DataFrame(E1E2).to_excel("E1E2 post.xlsx")  # . todo delete this

        # with masked array module from numpy. masked_inside is inclusive of endpoints
        mask = np.ma.masked_inside(E1E2, Epeak - epsilon, Epeak + epsilon + interp_step).mask

        # next is line 246 in the MATLAB code.
        dE = (x2i - y2i) / 2
        # pd.DataFrame(dE).to_excel("dE pre.xlsx")  # . todo delete this
        dE = dE.flatten()
        dE = dE[mask]  # keep only the values in the rectangle
        # pd.DataFrame(dE).to_excel("dE post.xlsx")  # . todo delete this

        # take the expected counts for the values in our rectangle
        expcnt = data2i.flatten()[mask]
        # pd.DataFrame(expcnt).to_excel("expcnt.xlsx")  # . todo delete this

        # add up counts for overlapped dE, check counts vs. reduced dE
        combo = np.array([dE, expcnt]).transpose()
        # pd.DataFrame(combo).to_excel("combo pre.xlsx")  # . todo delete this

        combo = combo[combo[:, 0].argsort()]  # sort based on the first column
        # pd.DataFrame(combo).to_excel("combo is=.2 sorted.xlsx")  # . todo delete this
        print("combo 1st,", combo[int(len(combo)/2-30):int(len(combo)/2+30)])
        # if neighboring rows are close enough, make them exactly close
        tol = 1e-3
        for k in range(len(combo[:, 0]) - 1):
            # row, column
            if abs(combo[k + 1, 0] - combo[k, 0]) < tol:
                # the next value is almost the same, make it the same
                combo[k + 1, 0] = combo[k, 0]
                # leave the expected counts alone
        # pd.DataFrame(combo).to_excel("combo combined.xlsx")  # . todo delete this
        # now we reduce the size of the combo list - histcounts section
        short_dE = np.unique(combo[:, 0])  # collect all unique values from dE
        # from matplotlib import pyplot as plt  # .
        # plt.plot(short_dE, label="short_dE")  # .
        new_cnts = np.zeros_like(short_dE)  # placeholder for the counts
        new_combo = np.array([short_dE, new_cnts]).transpose()
        # plt.plot(new_combo, label="new_combo")  # .
        # plt.figure()  # .
        combo_n = np.digitize(combo[:, 0], short_dE)  # create a mapping for which bin each value belongs
        # plt.plot(combo_n, ".", label="combo_n")  # .
        # plt.plot(combo, label="combo") # .
        print("\n combo", combo[int(len(combo)/2-30):int(len(combo)/2+30)])
        print("combo_n", combo_n[int(len(combo)/2-30):int(len(combo)/2+30)])  # . todo maybe try middle chunk???
        for i in range(len(combo[:, 0]) - 1):
            if abs(combo[:, 0][i] - combo[:, 0][i + 1]) < 1e5:  # . todo this only works because it basically says "if TRUE", but look into a better way and also look into if there might be a intentional reason for this way
                # for matching values, add the counts to the correct bin
                # print(i, combo[i,0], combo[i,1], combo_n[i], "|||", combo[:, 0][i], combo[:, 0][i + 1], combo[:, 0][i] - combo[:, 0][i + 1])
                # combo_n will identify the correct location, but it starts at 1.
                new_combo[:, 1][combo_n[i] - 1] += abs(combo[:, 1][i])
            else:  # . todo delete this else
                print("ELSED", i)
        print("new combo", new_combo[int(len(new_combo)/2-2):int(len(new_combo)/2+3)])
        # plt.plot(new_combo, ".", label="new_combo2")  # .
        # plt.legend()  # .
        # plt.show()  # .
        # normalize the data
        import sys  # . todo delete this and np.set statement and prints
        print("nc 1st", new_combo[:, 1])
        # np.set_printoptions(threshold=sys.maxsize)  # .
        C_norm = np.trapz(new_combo[:, 1], new_combo[:, 0])
        # print(sum(new_combo[:,1]))
        new_combo[:, 1] = new_combo[:, 1] / np.trapz(new_combo[:, 1],
                                                     new_combo[:, 0])  # order is opposite that of MATLAB

        # print("nc 2nd", new_combo[:, 1], np.trapz(new_combo[:, 1], new_combo[:, 0]))
        # shift the x-axis
        new_combo[:, 0] += 511
        # print("nc 3rd", new_combo[:,0])

        return new_combo, C_norm  # . added C_norm stuff <-/|\
