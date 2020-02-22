"""
***************************************************************************************************
custom_utils.clean_dataframe

This module contains customized utilities for cleaning data in pandas DataFrames:
    - preprocess_df (drops columns & null rows, creates new columns via concatenation,
      drops duplicates & returns column subset)
    - filter_dataframe (Filters pandas DataFrame according to contains expressions in a list)

Created on 12/31/19 by William Scardino
Last updated: 2/21/20
***************************************************************************************************
"""
def preprocess_df(df, new_col, con_col1, con_col2, subset_list, filter_col_list):
    """
    This function accepts a pandas DataFrame and it concatenates 2 columns into a new column.
    It also drops null rows from the pandas DataFrame.
    It returns a pandas DataFrame with specified columns defined by the user

    @param df: pandas DataFrame
    @param new_col: name of new column from concatenation result
    @param con_col1: name of first column to concatenate
    @param con_col2: name of second column to concatenate
    @param subset_list: list of columns to drop null values
    @param filter_col_list: list of columns to return in output DataFrame
    @ return: pre-processed pandas DataFrame containing output of newly concatenated pandas Series
    """
    # concatenate values of user-defined columns with a space between them to create a new pandas Series: df[new_col]
    df[new_col] = df[con_col1] + ' ' + df[con_col2]

    # drop null values from the DataFrame via a list of columns provided by the user: df
    df = df.dropna(subset=subset_list)

    # drop duplicates
    df = df.drop_duplicates(subset=[new_col])

    # return a subset of the DataFrame with only the columns specified by the user.
    df = df[filter_col_list]

    print('Preprocessed DataFrame!')
    print()
    print('New DataFrame Shape: {}'.format(df.shape))
    print()

    return df


def filter_dataframe(df, col_to_filter, value_to_filter):
    """
    Filters pandas DataFrame according to datetime parameter.
    Retains all data greater than or equal to user-defined year value.

    @param df: pandas DataFrame containing the input data
    @param col_to_filter: datetime year value that will be used to filter
    @param value_to_filter: datetime year value that will be used to filter that data according to "greater than or equal to" parameter test
    @return: filtered pandas DataFrame
    """
    # deep copy of DataFrame: df
    df = df.copy()

    # filter DataFrame col_to_filter according to value_to_filter
    df = df[df[col_to_filter] >= value_to_filter]

    print('Filtered out articles prior to {}!'.format(value_to_filter))
    print()
    print('DataFrame Shape: {}'.format(df.shape))
    print()

    return df

