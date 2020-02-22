"""
********************************************************************************************
core_utils.dataframe

This module contains common utilities for importing & handling data in pandas DataFrames:
    - union_csv (unions similar csv files into a single DataFrame)

created: 1/5/20
last updated: 2/21/20
********************************************************************************************
"""
import pandas as pd
from glob import glob
from os import path


def union_csv(csv_path: object, glob_pattern: object) -> object:
    """
    This function accepts a directory where csv files are located
    and a glob pattern for specific .csv naming conventions.
    It returns a single pandas DataFrame with the data from all the .csv files in the directory

    @param csv_path: csv_path: Raw string representation of a directory that contains .csv files
    @param glob_pattern: glob pattern specifying .csv file naming convention
    @return: unioned pandas DataFrame that contains all of the data from all the .csv files in the directory
    """
    # assign file path of directory containing csv files with raw string: csv_path
    csv_path = csv_path

    # assert that a raw string type was passed to csv_path parameter
    assert isinstance(csv_path, str), "You need to pass a raw string to csv_path!"

    # append all files with the glob pattern naming convention in the file path to a list: all_files
    all_files = glob(path.join(csv_path, glob_pattern))

    # read in each .csv file with pd.read_csv inside the for loop
    dfs = []

    for f in all_files:
        df = pd.read_csv(f)
        df['candidate'] = f.replace('data', '').replace('.', '').replace('//', '').split('_')[0]
        dfs.append(df)

    # ignore index of all DataFrames in memory and set sort parameter to False
    combined_df = pd.concat(dfs, ignore_index=True, sort=False)

    # print DataFrame first 5 rows, metadata, shape & number of rows per candidate
    print()
    print('DataFrame Sample')
    print(combined_df.head())
    print()
    print('DataFrame Shape: {}'.format(combined_df.shape))
    print()
    print('DataFrame Metadata')
    print(combined_df.info())
    print()
    print(df['candidate'].value_counts())
    print()

    # return pandas DataFrame utilizing a generator expression for the objs argument of pd.concat
    return combined_df
