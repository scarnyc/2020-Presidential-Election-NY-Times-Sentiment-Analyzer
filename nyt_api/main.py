"""
************************************************************************************************************************************
nyt_api.main

This module contains the get_articles() function for scraping data from the N.Y. Times Article Search API.
For more info on the N.Y. Times Article Search API please go to https://developer.nytimes.com/docs/articlesearch-product/1/overview

created: 12/31/19
last updated: 1/18/20
************************************************************************************************************************************
"""
import numpy as np
import requests
import time
from pandas import DataFrame
from pandas.io.json import json_normalize
import datetime as dt


def get_data(key, output_path, members):
    """
    This function accepts a N.Y. Times API key, the file directory where the csv files will go, and  list of members
    that an user wants to query the API for information; It outputs a csv file every call and the results are paginated
    in a loop that will call the API up to 4000 times.

    @param key: user's API key
    @param output_path: file directory to write csv files with NYT API results
    @param members: list of terms or subjects that will be used to query the API (list)
    """
    # check if members variable is list type
    assert isinstance(members, list), "You need to pass in a list to members!"

    # iterate over members list: member
    for member in members:
        # iterate api calls to paginate results: page
        for page in np.arange(0, 4000):
            # create base_url by joining the host url with member, predicates, page offset, and api key
            base_url = "".join(
                ['https://api.nytimes.com/svc/search/v2/articlesearch.json?q=',
                 member,
                 '&page=',
                 str(page),
                 '&sort=newest',
                 '&api-key=',
                 key]
            )
            print(base_url)
            # try the get request, if the call succeeds, return a DataFrame normalizing the json data
            try:
                r = requests.get(base_url)
                df: DataFrame = json_normalize(
                    data=r.json()['response'],
                    record_path='docs',
                    sep='_',
                    max_level=1
                )
                # pause for 6 seconds between each call
                time.sleep(6)
                # output csv files and include the current date in the naming convention
                df.to_csv(output_path + '/' + member + '_' + str(page) + str(0) + '_' + '{date:%Y.%m.%d}.csv' \
                          .format(date=dt.datetime.now())
                          )
            # if the call fails, raise the exception
            except Exception as e:
                raise e

# if __name__ == '__main__':
#     nyt_df = get_data(key=your_key,
#                       output_path=your_directory,
#                       members=['Bernie Sanders']
#                       )
