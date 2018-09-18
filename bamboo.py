"""
A library of functions that consume Pandas dataframes and return them refactored.

(In non-officially supported, but meaningful ways.)

It includes functions that manipulate data for specific reasons:
* load
* save
* rename
* convert
* reformat
* merge
* fit functions to data
* renumber data
* and many others...
"""
import pandas as pd
import os
import numpy as np
import collections as collect
from subprocess import run, call

##############################
### Single Value Functions ###
##############################


def cut_extension(fn):
    """Cut everything to first period L->R."""
    period = len(fn)
    for index in range(len(fn)-1, -1, -1):
        if fn[index] == '.':
            period = index
    retval = fn[:period]
    return retval


def not_float(val):
    """Boolean function to say if this value is not a float."""
    try:
        float(val)
        return False
    except ValueError:
        return True


def contains(string, substring):
    """Boolean function detecting if substring is part of string."""
    try:
        if string.index(substring) >= 0:
            return True
    except ValueError:
        return False


def find_index_of(string, character, occurrence=1):
    """Find the n-th index of a character in a string."""
    count = 0
    for char_index in range(len(string)):
        if string[char_index] == character:
            count += 1
            if count == occurrence:
                return char_index
    return -1


def remove_nans(array):
    """Remove nans from numpy array."""
    return array[~np.isnan(array)]


def match_list_entries(a, b):
    """
    Match up these two input lists by removing entries from one that correspond to nans or None in the other.
    Lists must be of same length.
    """
    indices_to_keep = []
    if len(a) != len(b):
        raise ValueError('Lists should be of same length!')
    exclude = [None, float('nan'), float('inf'), 'nan']
    for entry in range(len(a)):
        if a[entry] not in exclude and \
           b[entry] not in exclude and \
           not np.isnan(a[entry]) and \
           not np.isnan(b[entry]):
            indices_to_keep.append(entry)
    a = [a[i] for i in indices_to_keep]
    b = [b[i] for i in indices_to_keep]
    return a, b

######################
### File Functions ###
######################


def load_data(file_name, path, sep=',', add_ext=False, pandas_format=True, columns=None):
    """
    Primary function to load data.

    Will load any pandas csv. Optional separation indicator.
    Will load all types of objects, strings, floats, etc. not just floats.
    """
    if not add_ext:
        csv_path = os.path.join(path, file_name)
    else:
        csv_path = os.path.join(path, file_name+'.csv')
    if pandas_format:
        frame = pd.read_csv(csv_path, sep=sep, index_col=0, low_memory=False)
    else:
        if columns:  # no header row exists
            frame = pd.read_csv(csv_path, sep=sep,
                                low_memory=False, header=None, names=columns)
        else:  # header row exists
            frame = pd.read_csv(csv_path, sep=sep, low_memory=False)
    return frame


def save_data(frame, file_name, path, suffix='', cut_ext=False):
    """
    Primary function to save data to csv.

    If file name contains an extension, can be cut by setting cut_ext to True.
    """
    if cut_ext:
        name = cut_extension(file_name) + suffix + '.csv'
    else:
        name = file_name + suffix + '.csv'
    frame.to_csv(os.path.join(path, name), na_rep='None')


def xslx_to_csv(xlsx_name, csv_name, path):
    """Convert file at path from xlsx to csv."""
    # convert to csv
    # setup:
    #   RUN
    #   sudo apt install gnumeric
    #   sudo rm -f /etc/machine-id  # might be able to get away without doing this
    #   sudo dbus-uuidgen --ensure=/etc/machine-id
    flags = ['ssconvert', '-S', os.path.join(
        path, xlsx_name), os.path.join(path, csv_name)]
    run(flags)


def rename(old_name, new_name, old_path, new_path=None):
    """Rename one file with a new file name."""
    old_file = os.path.join(old_path, old_name)
    new_file = os.path.join(
        old_path, new_name) if new_path is not None else os.path.join(new_path, new_name)
    call(['mv', old_file, new_file])


######################################
### Data Frame Selection Functions ###
######################################


def where(frame, col, value):
    """Take slice of dataframe where col has value."""
    rows = []
    for row in range(frame.shape[0]):
        curr = frame.iloc[row][col]
        if curr == value:
            rows.append(row)
    frame = frame.iloc[rows]
    frame = renumber_index(frame)
    return frame


def where_partial(frame, col, value):
    """Take slice of dataframe where col partially contains value."""
    rows = []
    for row in range(frame.shape[0]):
        curr = frame.iloc[row][col]
        if contains(curr, value):
            rows.append(row)
    frame = frame.iloc[rows]
    frame = renumber_index(frame)
    return frame


def where_multi(frame, cols, values):
    """Take slice of dataframe where list of cols have corresponding values."""
    rows = []
    for row in range(frame.shape[0]):
        curr = list(frame.iloc[row][cols])
        if curr == values:
            rows.append(row)
    frame = frame.iloc[rows]
    frame = renumber_index(frame)
    return frame


def where_any(frame, col, values):
    """
    Take a slice of dataframe where col matches any of the values (list).

    If matches is False, the rows that match values are not included and those that do not are.
    It turns the function into an "all except these" where.
    """
    rows = []
    for row in range(frame.shape[0]):
        curr = frame.iloc[row][col]
        if curr in values:
            rows.append(row)
    frame = frame.iloc[rows]
    frame = renumber_index(frame)
    return frame


def remove_where(frame, col, values):
    """
    Take a slice of dataframe where col matches any of the values (list).

    If matches is False, the rows that match values are not included and those that do not are.
    It turns the function into an "all except these" where.
    """
    rows = []
    for row in range(frame.shape[0]):
        curr = frame.iloc[row][col]
        if curr not in values:
            rows.append(row)
    frame = frame.iloc[rows]
    frame = renumber_index(frame)
    return frame


def remove_partial(frame, col, value):
    """Take slice of dataframe where col contains value."""
    rows = []
    for row in range(frame.shape[0]):
        curr = frame.iloc[row][col]
        if not contains(curr, value):
            rows.append(row)
    frame = frame.iloc[rows]
    frame = renumber_index(frame)
    return frame

#########################################
### Data Frame Modification Functions ###
#########################################


def renumber_index(frame):
    """Renumbers the row indices of frame for proper .iloc[] indexing."""
    indices = pd.Series(list(range(frame.shape[0])))
    frame.index = indices
    return frame


def replace(frame, old, new, cols=None):
    """
    Replace all instances of 'old' with 'new' in the specified columns.

    Input: 
        * frame
        * old = value to replace
        * new = value to replace with
        * cols = LIST of columns to replace within. 
                 Defaults to all.
    """
    columns = cols if cols is not None else frame.columns
    for row in range(frame.shape[0]):
        for col in columns:
            if frame.iloc[row][col] == old:
                frame.loc[row, col] = new
    return frame


def functional_replace(frame, old, fun, cols=None):
    """
    Apply passed function to all values matching old in col.
    
    Input:
        * frame
        * old = value to replace
        * new = value to replace with
        * cols = LIST of columns to replace within.
                    Defaults to all.

    An example of this is to get global values for step numbers used across system,
    Instead of using string currently in place.
    """
    columns = cols if cols is not None else frame.columns
    for row in range(frame.shape[0]):
        for col in columns:
            if frame.loc[row, col] == old:
                frame.loc[row, col] = fun(old)
    return frame


def convert_to_numeric(frame):
    """
    Convert this dataset to a numeric-typed dataset.

    Input: A dataset that contains only numbers but has fields labeled as objects or other.
    Output: The same dataset, retyped to appropriate numerics (floats, ints, etc).
    """
    for column in frame.columns:
        frame.loc[:, column] = pd.to_numeric(
            frame.loc[:, column], errors='coerce')
    return frame


def reorder_columns(frame, front_columns):
    """
    Re-order the columns of frame placing front_columns first.
    
    Good for looking specifically at certain columns for specific file in pipeline.
    """
    # reorder columns into appropriate order
    new_cols = list(frame.columns)
    for d in front_columns:
        new_cols.remove(d)
    new_cols = front_columns + new_cols
    frame = frame[new_cols]
    return frame


def drop_column_suffix(frame, suffix):
    """Drop this suffix from each column that contains it."""
    frame.columns = [col if not col.endswith(
        suffix) else col[:-len(suffix)] for col in frame.columns]
    return frame


def add_column(frame, column, fill_value=None):
    """Take folder and add fill_value category of that value to each row."""
    if column not in list(frame.columns):
        frame[column] = fill_value
    return frame


def slice_column_values(frame, column, start=0, end=None):
    """
    Take frame and trim every value in the column specified by the number of indices specified.
    
    Take specified slice of each column.
    The default values of start and end do not change the length of any of the values.
    """
    old_series = frame[column]
    new_series = []
    for row in range(old_series.shape[0]):
        curr = old_series[row]
        piece = curr[start:end] if end is not None else curr[start:len(curr)]
        new_series.append(piece)
    new_series = pd.Series(new_series, name=column)
    frame[column] = new_series
    return frame


def isolate_before_nans(frame, attr):
    """Return a dataset that does not contain nans at the end."""
    for row in range(frame.shape[0]-1, 0, -1):
        if frame.iloc[row][attr] != float('nan'):
            return frame.loc[:row, :]
    return frame  # (empty)

############################
### Inter-File Functions ###
############################


def combine_files(suffix, folders, path):
    """
    
    From each folder, take each file with this suffix, load it and append it to the resultant frame.

    Input:
        suffix = End of file name, before extension
        folders = list of folder to collect files from
        path = root path to all folders referenced
    Output:
        Saved file labeled 'dataset_'+suffix, containing the dataframe of all frames appended.
        Data from each folder is labeled with a column 'folder' with the folder name embedded.
    Returns:
        Frame that is saved.

    Suffix should not include .csv
    Only .csv files should be specified.
    """
    frame = None
    for folder in folders:
        folder_path = os.path.join(path, folder)
        try:
            df = load_data(folder + suffix, folder_path, add_ext=True)
            df = add_column(df, 'folder', folder)
            frame = frame.append(df, sort=True) if frame is not None else df
        except FileNotFoundError:
            print('\tskipping', folder)
            continue
    save_data(frame, 'dataset'+suffix, path, suffix)
    return frame


###############################
### Preprocessing Functions ###
###############################

def combine_rows(frame, on, suffix_attr):
    """
    Join rows into one row based on category.
    
    Input:
        * frame = pd.DataFrame this is about
        * on = attribute that each row that will be combined shares in value
        * suffix_attr = attribute that differentiates rows to be combined with each other
    Returns:
        * joined data frame
    """
    joined_df = None
    for row in range(frame.shape[0]):
        curr = frame.loc[row, :].to_frame('curr').transpose()
        suffix = str(curr.iloc[0][suffix_attr])
        if joined_df is not None:
            joined_df = pd.merge(
                joined_df, curr, on=on, suffixes=('', '_'+suffix))
        else:
            curr.columns = [str(col) + '_' + suffix if str(col)
                            != on else str(col) for col in curr.columns]
            joined_df = curr
    step_cols = [
        col for col in joined_df.columns if contains(col, suffix_attr)]
    joined_df = joined_df.drop(step_cols, axis=1)
    return joined_df


################################
##### Fitting Functions ########
################################
import copy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
# remove rankwarning from polyfit b/c it's slow
import warnings
warnings.simplefilter('ignore', np.RankWarning)


# general form functions to optimize with curve_fit
def linear(t, a, b):  # 1st order polynomial
    return a*t + b


def quad(t, a, b, c):  # 2nd order polynomial
    return a*t**2 + b*t + c


def poly(t, a, b, c, d):  # 3rd order polynomial
    return a*t**3 + b*t**2 + c*t + d


def log(t, a, b):  # logarithmic
    return a + b * np.log(t)


def exp(t, a, b, c):  # single exponential
    return a * np.exp(b * t) + c


def exp2(t, a, b, c, d):  # pair of exponentials
    return a*np.exp(b*t) + c*np.exp(d*t)


def get_y_fit(x, fit, fit_type):
    y_fit = []
    for i_ in range(len(x)):
        t = x[i_]
        if fit_type == 'mean':
            y_fit.append(fit[0])
        elif fit_type == 'linear':
            y_fit.append(linear(t, fit[0], fit[1]))
        elif fit_type == 'quad':
            y_fit.append(quad(t, fit[0], fit[1], fit[2]))
        elif fit_type == 'poly':
            y_fit.append(poly(t, fit[0], fit[1], fit[2], fit[3]))
        elif fit_type == 'log':
            y_fit.append(log(t, fit[0], fit[1]))
        elif fit_type == 'exp':
            y_fit.append(exp(t, fit[0], fit[1], fit[2]))
        elif fit_type == 'exp2':
            y_fit.append(exp2(t, fit[0], fit[1], fit[2], fit[3]))
        else:
            raise ValueError('not valid fit model')
    return np.array(y_fit)


def get_rsquared(x, y, fit, fit_type):
    """
    Using x values compare y values to fit(x) values.
    
    x and y should be np.arrays
    SST = Sum(i=1..n) (y_i - y_bar)^2
    SSReg = Sum(i=1..n) (y_ihat - y_bar)^2
    Rsquared = SSReg/SST

    https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
    """
    y_bar = y.mean()  # should work b/c y is np array
    y_fit = get_y_fit(x, fit, fit_type)
    sstot = sum((y - y_bar)**2)  # total sum of squares
    ssres = sum((y - y_fit)**2)  # squares of residuals
    return 1 - (ssres / sstot)


def get_mse(x, y, fit, fit_type):
    """Get mean squared error on true y and y_hat from specified fit."""
    y_fit = get_y_fit(x, fit, fit_type)
    mse = ((y - y_fit)**2).mean()  # squares of residuals
    return mse


def get_fits(x, y, fit_types):
    """
    Input: 
        x = array of cycle numbers
        y = array of corresponding values at those time points

    Return list of each type of fit for this step/var."""
    fits = []
    functions = {  # reference dictionary for actual math functions
        'linear': linear,
        'quad': quad,
        'poly': poly,
        'log': log,
        'exp': exp
    }

    def get_fit_of_this_type(x, y, fit_type):
        func = functions[fit_type]
        if fit_type == 'exp':
            try:
                popt, pcov = curve_fit(func, x,  y)  # , p0=(approx))
            except RuntimeError:
                return None
        else:
            popt, pcov = curve_fit(func, x, y)
        mse = get_mse(x, y, popt, fit_type)
        r2 = get_rsquared(x, y, popt, fit_type)
        return [popt, pcov, fit_type, mse, r2]

    ### begin ###
    if 'log' in fit_types:
        y[y == 0] = 1e-10
    for fit_type in fit_types:  # , 'exp2']
        fit = get_fit_of_this_type(x, y, fit_type)
        if fit is not None:
            fits.append(fit)
    return fits


def fit_data(frame, uniqueness_order, uniqueness_dict, x_cols, y_cols, fit_types=['linear', 'quad', 'poly', 'log', 'exp']):
    """
    Recursive function to collapse each column into a row of up to eight column variables depending on fit_type requested.

    If multiple fit_types are requested, the function will return the optimal fit based on the R^2 metric.

    Returned frame will include the fit parameter and its stadard deviation.
    """
    def match_up(x, y):
        difference = len(x) - len(y)
        y = y.tolist()
        x = x.tolist()
        if difference > 0:  # timesubset bigger
            for i in range(difference):
                y.append(0.0)
        else:
            for i in range(difference):
                x.append(x[-1:])
        y = np.array(y)
        x = np.array(x)
        difference = len(x) - len(y)
        return x, y
    # recursive function
    if len(x_cols) != 1 and len(x_cols) != len(y_cols):
        raise ValueError(
            "X_cols and Y_cols don't match up! Incorrect specification.")
    ## BASE CASE ##
    if not uniqueness_order:
        fit = collect.defaultdict(list)
        # represents (minus the uniqueness the rows)
        frame = convert_to_numeric(frame)
        for index in range(len(y_cols)):
            y_col = y_cols[index]
            x_col = x_cols[0] if len(x_cols) == 1 else x_cols[index]
            x = frame[x_col].values.flatten()
            y = frame[y_col].values.flatten()
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            if len(x) != len(y):
                x, y = match_up(x, y)
            if len(y) == 0:
                continue
            # determine weights
            x = x
            y = y
            fits = get_fits(x, y, fit_types)

            for popt, pcov, fit_type, mse, r2 in fits:
                a, a_std = popt[0], pcov[0, 0]
                b, b_std = popt[1] if len(
                    popt) > 1 else 0.0, pcov[1, 1] if len(popt) > 1 else 0.0
                c, c_std = popt[2] if len(
                    popt) > 2 else 0.0, pcov[2, 2] if len(popt) > 2 else 0.0
                d, d_std = popt[3] if len(
                    popt) > 3 else 0.0, pcov[3, 3] if len(popt) > 3 else 0.0
                fit['fit_desc'] = [
                    'fit_type', 'mse', 'r2', 'a', 'a_std', 'b', 'b_std', 'c', 'c_std', 'd', 'd_std'
                ]
                fit[y_col+'_fit'] = [
                    fit_type, mse, r2, a, a_std, b, b_std, c, c_std, d, d_std
                ]
        fit = pd.DataFrame(fit)
        return fit

    # recursive case!
    level = uniqueness_order[0]  # key to work with
    level_values = uniqueness_dict[level]  # values to recurse on
    general_fit = None
    for level_value in level_values:
        subset = where(frame, level, level_value)
        fit = curve_fit(subset, uniqueness_order[1:],
                        uniqueness_dict, x_cols, y_cols, fit_types)
        fit = add_column(fit, level, fill_value=level_value)
        general_fit = fit if general_fit is None else general_fit.append(fit)
    return general_fit


def normalize_dataset(frame):
    """
    Normalize frame between [-1, 1] using StandardScalar from sklearn.
    
    Note: In machine learning context, this should be done on train, validation, and test separately.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    frame = pipeline.fit_transform(frame)
    return frame
