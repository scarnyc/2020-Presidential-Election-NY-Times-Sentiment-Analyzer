"""
***************************************************************************************************
model_utils.feature_eng

This package contains customized utilities for engineering features for Sentiment Analysis models:
    - date_feats (generates date features for model)
    - my_stopwords (dictionary of stopwords for model preprocessing)
    - tb_sentiment (generates sentiment and subjectivity scores for labeling)
    - nltk_sentiment (generates sentiment scores for labeling)
    - row_avg (computes the avg value across row values for 2 columns in a pandas DataFrame)
    - sentiment_label (generates the labels for sentiment analysis: ['positive','neutral','negative']
    - char_count (counts the number of characters in a text string)
    - split_x_y (splits X and y variables into training & test sets for machine learning)

created: 12/31/19
last updated: 1/5/20
***************************************************************************************************
"""
import pandas as pd
import numpy as np
import spacy
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# load spacy NLP model: nlp
nlp = spacy.load('en_core_web_md')

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


def date_feats(df, date_col):
    """
    This function generates new date features from an existing date Series in a pandas DataFrame.
    It returns the pandas DataFrame passed in by the user with new date features.

    args:
        - df: pandas DataFrame
        - date_col: name of date column used to generate new features
    reqs:
        import pandas as pd
    returns:
        pandas.DataFrame
    """
    # convert df[date_col] to datetime data type: df[date_col]
    df[date_col] = pd.to_datetime(df[date_col])
    # scrape month from df[date_col]: df['month']
    df['month'] = df[date_col].dt.month
    # scrape day from df[date_col]: df['day']
    df['day'] = df[date_col].dt.day
    # scrape dayofweek from df[date_col]: df['dayofweek']
    df['dayofweek'] = df[date_col].dt.dayofweek
    # scrape hour from df[date_col]: df['hour']
    df['hour'] = df[date_col].dt.hour
    # return df
    return df


def tb_sentiment(text):
    """
    This function generates sentiment labels from an existing text Series.
    It returns the sentiment and subjectivity scores generated by TextBlob.

    args:
        - text: text to be scored by TextBlob
    reqs:
        from textblob import TextBlob
    returns:
        TextBlob(text).sentiment
    """
    return TextBlob(text).sentiment


def nltk_sentiment(text):
    """
    This function generates sentiment labels from an existing text Series.
    It returns the sentiment and subjectivity scores generated by SentimentIntensityAnalyzer.

    args:
        - text: text to be scored by SentimentIntensityAnalyzer
    reqs:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    returns:
        SentimentIntensityAnalyzer().polarity_scores(text)
    """
    return SentimentIntensityAnalyzer().polarity_scores(text)


def row_avg(df, col1, col2, avg_col):
    """
    This function computes the mean across the row values of two columns in a pandas DataFrame.
    It returns a DataFrame containing the row average of the values in the two columns.

    args:
        - df: pandas DataFrame
        - col1: name of first column used to compute mean
        - col2: name of second column used to compute mean
        - avg_col: name of new column that will hold row mean
    reqs:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    returns:
        pandas.DataFrame
    """
    df[avg_col] = df[[col1, col2]].mean(axis=1)
    return df


def sentiment_label(df, col_for_label, label_col):
    """
    This function generates the labels for sentiment analysis: ['positive','neutral','negative']
    by utilizing a np.where function.
    It returns a DataFrame containing the the new label column.

    np.where logic:
        - if df[col_for_label] >= 0.05, return 'positive'
        - otherwise if df[col_for_label] is greater than -0.05 & less than 0.05, return 'neutral'
        - otherwise if df[col_for_label] <= -0.05, return 'negative'
        - else, return np.nan
    args:
        - df: pandas DataFrame
        - col_for_label: name of column to be used to generate labels
        - label_col: name of column that will hold newly generated labels
        import numpy as np
    returns:
        pandas.DataFrame
    """
    df[label_col] = np.where(
        df[col_for_label] >= 0.05,
        'positive',
        np.where(
            (df[col_for_label] > -0.05) & (df[col_for_label] < 0.05),
            'neutral',
            np.where(
                df[col_for_label] <= -0.05,
                'negative',
                np.nan
            )
        )
    )
    return df


def char_count(df, text, new_pd_series):
    """
    This function counts the number of characters per row of text in a DataFrame series.
    The function generates a new pandas Series containing the number of characters per row.
    It returns a DataFrame containing the new column.

    params:
        - df: pandas DataFrame
        - text: name of column used to count characters
        - new_pd_series: name of column that will hold newly generated counts
    returns:
        pandas.DataFrame
    """
    df[new_pd_series] = df[text].apply(len)

    return df


def lemma_nopunc(text):
    """
    This function lemmatizes tokens and removes punctuation from a string based on the spaCy NLP medium model.

    @param text: string to be lemmatized and stripped of punctuation
    @return: returns cleaned string.
    """

    # lemmatize tokens: lemmas
    lemmas = [token.lemma_ for token in nlp(text)
              if token.is_alpha]

    # Remove punctuation: no_punc
    no_punc = ' '.join(re.sub(r'[^\w\s]', '', t) for t in lemmas)

    return no_punc


def split_x_y(df, features, label, contains_col, contains_term, mapper):
    """
    Splits dataframe into training and testing sets for features & labels.

    @param mapper:
    @param label:
    @param features:
    @param df:
    @param contains_col:
    @param contains_term:
    @return:
    """
    X_train = df[~df[contains_col].str.contains(contains_term, case=False)][features]
    X_test = df[df[contains_col].str.contains(contains_term, case=False)][features]
    y_train = df[~df[contains_col].str.contains(contains_term, case=False)][label].map(mapper)
    y_test = df[df[contains_col].str.contains(contains_term, case=False)][label].map(mapper)

    return X_train, X_test, y_train, y_test