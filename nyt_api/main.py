"""
************************************************************************************************************************************
nyt_api.main

This package contains the function for scraping data by calling the N.Y. Times Article Search API:
    - get_articles (calls & queries API by user-defined value)

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

    args:
        - key: user's API key
        - output_path: file directory to write csv files with NYT API results
        - members: list of terms or subjects that will be used to query the API (list)
    reqs:
        - pandas
        - glob.glob
        - os.path
    """
    # check if members variable is list type
    assert type(members) == list, "You need to pass in a list to members!"

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

if __name__ == '__main__':
    get_data(key='ogATnoEYDEO64smhyKzgJ9H4Z4arnvxX',
             output_path=r'C:\Users\billy\PycharmProjects\nyt_sentiment_analyzer\data',
             members=['Donald Trump', 'Elizabeth Warren', 'Kamala Harris'
                      'Joe Biden', 'Andrew Yang', 'Tom Steyer', 'Pete Buttigieg', 'Michael Bloomberg', 'Tulsi Gabbard']
             )