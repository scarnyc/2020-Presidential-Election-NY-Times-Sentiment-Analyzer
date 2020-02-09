"""
********************************************************************************************
core_utils.dataframe

This package contains common utilities for importing & handling data in pandas DataFrames:
    - union_csv (unions similar csv files into a single DataFrame)
********************************************************************************************
"""
import pandas as pd
from glob import glob
from os import path


def union_csv(csv_path: object, glob_pattern: object) -> object:
    """
    This function accepts a directory where csv files are located
    and a glob pattern for specific .csv naming conventions.
    It returns a single pandas DataFrame with the contents from all the files in the directory.

    args:
        - csv_path: Directory that contains .csv files (raw str)
        - glob_pattern: Glob pattern specifying .csv file naming convention
    reqs:
        - pandas
        - glob.glob
        - os.path
    returns:
        pandas.DataFrame
    """
    # assign file path of directory containing csv files with raw string: csv_path
    csv_path = csv_path

    # assert that a raw string type was passed to csv_path parameter
    assert type(csv_path) == str, "You need to pass a raw string to csv_path!"

    # append all files with the glob pattern naming convention in the file path to a list: all_files
    all_files = glob(path.join(csv_path, glob_pattern))

    # read in each .csv file with pd.read_csv inside the generator expression
    # ignore index of all DataFrames in memory and set sort parameter to False
    combined_df = pd.concat(
        objs=(pd.read_csv(f) for f in all_files),
        ignore_index=True,
        sort=False
    )

    print('DataFrame Sample')
    print(combined_df.head())
    print()
    print('DataFrame Shape: {}'.format(combined_df.shape))
    print()

    # return pandas DataFrame utilizing a generator expression for the objs argument of pd.concat
    return combined_df
