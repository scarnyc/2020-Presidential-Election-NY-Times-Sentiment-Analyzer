"""
***************************************************************************************************
custom_utils.clean_dataframe

This package contains customized utilities for cleaning data in pandas DataFrames:
    - preprocess_df (drops columns & null rows, creates new columns via concatenation,
    drops duplicates & returns column subset)

created: 12/31/19
last updated: 1/1/20
***************************************************************************************************
"""
def preprocess_df(df, new_col, con_col1, con_col2, subset_list, filter_col_list):
    """
    This function accepts a pandas DataFrame and it concatenates 2 columns into a new column.
    It also drops null rows from the pandas DataFrame.
    It returns a pandas DataFrame with specified columns defined by the user.

    args:
        - df: pandas DataFrame
        - new_col: name of new column from concatenation result
        - con_col1: name of first column to concatenate
        - con_col2: name of second column to concatenate
        - subset_list: list of columns to drop null values
        - filter_col_list: list of columns to return in output DataFrame
    returns:
        pandas.DataFrame
    """
    # concatenate values of user-defined columns with a space between them to create a new pandas Series: df[new_col]
    df[new_col] = df[con_col1] + ' ' + df[con_col2]

    # drop null values from the DataFrame via a list of columns provided by the user: df
    df = df.dropna(subset=subset_list)

    # drop duplicates
    df = df.drop_duplicates()

    # return a subset of the DataFrame with only the columns specified by the user.
    df = df[filter_col_list]

    print('Preprocessed DataFrame!')
    print()
    print('New DataFrame Shape: {}'.format(df.shape))
    print()

    return df
