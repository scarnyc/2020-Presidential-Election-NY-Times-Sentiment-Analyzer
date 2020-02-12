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
last updated: 1/4/20
***************************************************************************************************
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from nltk.probability import FreqDist
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

# define stopwords variable: my_stopwords
my_stopwords = {
    'a',
    'about',
    'above',
    'after',
    'again',
    'all',
    'also',
    'am',
    'an',
    'and',
    'any',
    'are',
    'as',
    'at',
    'be',
    'because',
    'been',
    'before',
    'being',
    'below',
    'between',
    'both',
    'br',
    'but',
    'by',
    'can',
    'com',
    'could',
    'did',
    'do',
    'does',
    'doing',
    'down',
    'during',
    'each',
    'else',
    'ever',
    'few',
    'film',
    'films',
    'for',
    'from',
    'further',
    'get',
    'had',
    'has',
    'have',
    'having',
    'he',
    "he'd",
    "he'll",
    "he's",
    'her',
    'here',
    "here's",
    'hers',
    'herself',
    'him',
    'himself',
    'his',
    'how',
    "how's",
    'however',
    'http',
    'i',
    "i'd",
    "i'll",
    "i'm",
    "i've",
    'if',
    'in',
    'into',
    'is',
    'it',
    "it's",
    'its',
    'itself',
    'just',
    'k',
    "let's",
    'like',
    'me',
    'more',
    'most',
    'movie',
    'movies',
    'my',
    'myself',
    'of',
    'off',
    'on',
    'once',
    'only',
    'or',
    'other',
    'otherwise',
    'ought',
    'our',
    'ours',
    'ourselves',
    'out',
    'over',
    'own',
    'r',
    'same',
    'shall',
    'she',
    "she'd",
    "she'll",
    "she's",
    'should',
    'since',
    'so',
    'some',
    'such',
    'than',
    'that',
    "that's",
    'the',
    'their',
    'theirs',
    'them',
    'themselves',
    'then',
    'there',
    "there's",
    'these',
    'they',
    "they'd",
    "they'll",
    "they're",
    "they've",
    'this',
    'those',
    'through',
    'to',
    'too',
    'under',
    'until',
    'up',
    'very',
    'was',
    'watch',
    'we',
    "we'd",
    "we'll",
    "we're",
    "we've",
    'were',
    'what',
    "what's",
    'when',
    "when's",
    'where',
    "where's",
    'which',
    'while',
    'who',
    "who's",
    'whom',
    'why',
    "why's",
    'with',
    'would',
    'www',
    'you',
    "you'd",
    "you'll",
    "you're",
    "you've",
    'your',
    'yours',
    'yourself',
    'yourselves'
}


def get_word_freq(pd_series, n=None):
    """
    List the words in a vocabulary according to number of occurrences in a text corpus.

    Dependencies: instantiation of cv (cell above)
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) ->
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]

     args:
        - pd_series: pandas Series used to generate word frequencies
        - n: essentially top N, number of tokens to generate that are most frequent
    returns:
        pandas.DataFrame
    """
    # Build the vectorizer: vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=my_stopwords)

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


def plot_word_freq(pd_series, plot_title, n=None):
    """
    List the words in a vocabulary according to number of occurrences in a text corpus via a plotly bar graph.

    @param pd_series: pandas Series used to generate word frequencies
    @param plot_title: Title of bar graph
    @param n: essentially top N, number of tokens to generate that are most frequent
    @return: plotly.offline.fig
    """
    # call the get_word_freq to get top N occurring features in a corpus: common_words
    common_words = get_word_freq(pd_series, n=n)
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
    fig = go.Figure(data=[go.Bar(
        x=count_df['freq'],
        y=count_df['word'],
        text=count_df['word'],
        textposition='auto',
        orientation='h'
    )])

    # Customize aspect
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5,
                      opacity=0.6
                      )

    # customize layout
    fig.update_layout(
        title_text=plot_title,
        yaxis=dict(autorange="reversed",
                   tickvals=count_df['word'])
        # ,
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)'
    )
    # write the figure
    fig.write_image("images/word_freq_bar_graph_eda.png")
    print('saved Bar Graph .png!')
    print()


def two_dim_tf_viz(df, pd_series, pd_color_series, pd_hover_series, plot_title, max_features=None, random_state=42):
    """
    Plots a scatter plot of Term Frequency, Inverse Document Frequency Ratios transformed to a 2d space, via plotly.

    @param df: first df that has pandas Series text data
    @param pd_series: pandas text Series used to generate tfidf scores
    @param max_features: max number of features for TfidfVectorizer (default=None)
    @param pd_color_series: pandas Series used to color Scatter plot
    @param pd_hover_series: pandas Series used to include in mouse hover
    @param plot_title: title of Scatter plot
    @param random_state: random seed variable for SVD model
    @return: plotly.offline.fig
    """
    # Build the vectorizer, specify max features
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stopwords, max_features=max_features)

    # Fit and transform
    tfidf.fit_transform(df[pd_series])

    # Transform
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

    # create Scatter Plot
    fig = px.scatter(
        svd_df,
        x='PC1',
        y='PC2',
        color=pd_color_series,
        hover_data=[pd_hover_series]
    )
    # update trace list
    fig.update_traces(
        marker=dict(size=12,
                    line=dict(width=2,
                              color='DarkSlateGrey')
                    ),
        selector=dict(mode='markers')
    )
    # update layout
    fig.update_layout(title_text=plot_title
                      # ,
                      # paper_bgcolor='rgba(0,0,0,0)',
                      # plot_bgcolor='rgba(0,0,0,0)'
                      )
    # write the figure
    fig.write_image("images/scatter plot.png")

    print('saved Scatter Plot .png!')
    print()


def time_series_line_viz(df, date_index, pd_series, plot_title):
    """
    Plots a line plot over a Daily Time Series via plotly

    @param df: DataFrame containing Time Series data
    @param date_index: Datetime index
    @param pd_series: pandas Series to plot over time
    @param plot_title: Title of line plot
    @return: plotly.offline.fig
    """
    # Set the index of ds_tweets to pub_date
    date_df = df.set_index(date_index).copy()

    # Generate average sentiment scores for #python
    sentiment = date_df[pd_series].resample('1 d').mean()

    # Create trace
    fig = go.Figure()

    # add trace for Line Plot
    fig.add_trace(
        go.Scatter(
            x=sentiment.index,
            y=sentiment,
            mode='lines+markers',
            name='lines+markers',
            marker_color='rgb(49,130,189)'
        ))

    # update layout
    fig.update_layout(title_text=plot_title)

    # write the figure
    fig.write_image("images/line plot.png")

    print('saved Line Plot .png!')
    print()


def corr_heatmap(df, cols):
    """

    @param df:
    @param cols:
    """
    # print DataFrame summary statistics
    print('DataFrame descriptive statistics!')
    print(df.describe())
    print()

    # Create the correlation matrix
    corr = df.copy()[cols].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Add the mask to the heatmap
    sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f")

    # show and save image
    plt.show()
