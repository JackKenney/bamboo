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

Author: Jack Kenney
Copyright 2018
MIT License

Version 1.0.0
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


def is_float(val):
    """Boolean function to say if this value is not a float."""
    try:
        float(val)
        return True
    except ValueError:
        return False


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
