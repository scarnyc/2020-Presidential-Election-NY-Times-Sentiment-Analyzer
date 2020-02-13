"""
***************************************************************************************************
graph_utils.graph

This package contains customized utilities for performing graphical EDA:
    - my_stopwords (dictionary of stopwords for model pre-processing)
    - get_word_freq (List the words in a vocabulary according to number of occurrences in a text corpus)
    - plot_word_freq (Plot the word frequencies via a plotly Bar Graph)
    - two_dim_tf_viz (Plots text tf-idf scores reduced to 2 dim. via a plotly Scatter plot)
    - time_series_line_viz (Plots a line plot over a Daily Time Series via plotly)
    - corr_heatmap (Plots a Heatmap between the correlation of variables)

created: 12/31/19
last updated: 2/13/20
***************************************************************************************************
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Set the default seaborn style
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
# import plotly.express as px
# import plotly.graph_objs as go


def get_word_freq(pd_series, stopwords, n=None):
    """
    List the words in a vocabulary according to number of occurrences in a text corpus.

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]

    @param pd_series: text feature used to generate Bag of Words model
    @param stopwords: List of Stopwords to filter out in CountVectorizer
    @param n: top N features based on word frequency
    @return: List of top N words & frequencies tuples
    """
    # Build the vectorizer: vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=stopwords)

    # Fit and transform pd_series: bow
    bow = vectorizer.fit_transform(pd_series)

    # Create the bow DataFrame representation: bow_df
    bow_df = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())

    # sum word frequency across all documents: bow_df
    sum_words = bow.sum(axis=0)

    # get word, frequency tuples across all words: words_freq
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    # sort tuples by frequency in descending order: words_freq
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # print word freq
    print(bow_df.sum().sort_values(ascending=False))
    print()

    return words_freq[:n]


def plot_word_freq(pd_series, plot_title, stopwords, n=None):
    """
    List the words in a vocabulary according to number of occurrences in a text corpus via a plotly bar graph.

    @param pd_series: pandas Series used to generate word frequencies
    @param plot_title: Title of bar graph
    @param stopwords: Stopwords to use with vectorizer
    @param n: essentially top N, number of tokens to generate that are most frequent
    """
    # call the get_word_freq to get top N occurring features in a corpus: common_words
    common_words = get_word_freq(pd_series, stopwords=stopwords, n=n)
    # instantiate empty lists that will be used by for loop
    count_dfs = []
    count_list = []

    # iterate over common_words with a word, freq tuple
    for word, freq in common_words:
        # append each tuple to count_list
        count_list.append((word, freq))
        # create a DataFrame using count_list
        df = pd.DataFrame(count_list)
        # append the DataFrames in count_dfs
        count_dfs.append(df)

    # concatenate dataframes
    count_df = pd.concat(count_dfs)

    # dedupe
    count_df.drop_duplicates(inplace=True)

    # name columns
    count_df.columns = ['word', 'freq']

    # filter out 'pron'
    count_df = count_df.loc[~count_df['word'].str.contains('pron', case=False)]

    # visualize the data
    sns.barplot(data=count_df,
                y=count_df['word'],
                x=count_df['freq'],
                kind='h',
                color='blue'
                )

    # add title to bar plot
    plt.title(plot_title)

    # show the plot
    plt.show()


def two_dim_tf_viz(df, pd_series, pd_color_series, plot_title, stopwords, max_features=None, random_state=42):
    """
    Plots a scatter plot of Term Frequency, Inverse Document Frequency Ratios transformed to a 2d space, via plotly.

    @param df: first pandas DataFrame that contains text Series
    @param pd_series: pandas text Series used to generate tfidf scores
    @param max_features: max number of features for TfidfVectorizer (default=None)
    @param pd_color_series: pandas Series used to color Scatter plot
    @param plot_title: title of Scatter plot
    @param stopwords: stopwords to use with vectorizer
    @param random_state: random seed variable for SVD model
    """
    # Build the vectorizer, specify max features
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords, max_features=max_features)

    # Fit TfidfVectorizer
    tfidf.fit(df[pd_series])

    # Transform BOW matrix
    tf_bow = tfidf.transform(df[pd_series])

    # instantiate & fit SVD model: svd
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    svd.fit_transform(tf_bow)

    # reduce the features to 2D: reduced_features_df
    reduced_features_df = svd.transform(tf_bow)

    # join SVD features with df2: svd_df
    svd_df = df.join(pd.DataFrame(reduced_features_df, columns=['PC1', 'PC2']))
    print(svd_df.info())
    print()

    # Define a custom continuous color palette
    color_palette = sns.light_palette('orangered',
                                      as_cmap=True)

    # Plot mapping the color of the points with custom palette
    sns.scatterplot(x='PC1',
                    y='PC2',
                    hue=pd_color_series,
                    data=svd_df,
                    palette=color_palette)

    # add title to scatter plot
    plt.title(plot_title)

    # show the plot
    plt.show()


def time_series_line_viz(df, date_index, pd_series, plot_title):
    """
    Plots a line plot over a Daily Time Series via plotly

    @param df: pandas DataFrame containing Time Series data
    @param date_index: pnadas Datetime index
    @param pd_series: pandas Series to plot mean aggregations over time
    @param plot_title: Title of Line Plot to display
    """
    # Set the index of ds_tweets to pub_date
    date_df = df.set_index(date_index).copy()

    # Generate average sentiment scores for #python
    sentiment = date_df[pd_series].resample('1 d').mean()

    # create subplots
    fig, ax = plt.subplots()

    # Add the time-series for "relative_temp" to the plot
    ax.plot(sentiment.index, sentiment)

    # Set the x-axis label
    ax.set_xlabel('Time')

    # Set the y-axis label
    ax.set_ylabel('Average Sentiment')

    # add title
    plt.title(plot_title)

    # Show the figure
    plt.show()


def corr_heatmap(df, features):
    """
    This function plots a Heatmap that displays correlations between different features for EDA

    @param df: pandas DataFrame that contains features
    @param features: list of features to be used to generate correlation Heatmap
    """
    # print DataFrame summary statistics
    print('DataFrame descriptive statistics:')
    print(dict(df.describe()))
    print()

    # Create the correlation matrix
    corr = df.copy()[features].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Add the mask to the heatmap
    sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f")

    # show and save image
    plt.show()
