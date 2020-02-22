"""
**************************************************************************************************************
model_utils.feature_eng

This module contains customized utilities for engineering features for Sentiment Analysis models:
    - date_feats (generates date features for model)
    - my_stopwords (dictionary of stopwords for model pre-processing)
    - tb_sentiment (generates sentiment and subjectivity scores for labeling)
    - sentiment_analyzer (applies sentiment labels row-wise using a text-based column feature)
    - sentiment_label (generates the labels for sentiment analysis: ['positive','neutral','negative']
    - custom_label(change labels for negative sentiment if a user-selected column contains user-defined terms)
    - char_count (counts the number of characters in a text string)
    - apply_func (apply a function to a pandas series (row-wise) and return the resulting DataFrame)
    - drop_high_corr (Drop highly correlated features from a DataFrame)
    - get_vocab_size (retreive a count of unique words in a vocabulary)

Created on 12/31/19 by William Scardino
Last updated: 2/21/20
**************************************************************************************************************
"""
import pandas as pd
import numpy as np
from spacy.lang.en import English
import re
from textblob import TextBlob

# load spacy English model: nlp
nlp = English()


def lemma_nopunc(text):
    """
    Lemmatize tokens & remove punctuation from a string based on the spaCy NLP English() model.

    @param text: string to be lemmatized and stripped of punctuation
    @return: returns cleaned string.
    """

    # lemmatize tokens: lemmas
    lemmas = [token.lemma_ for token in nlp(str(text))
              if token.is_alpha and token.lemma_ != '-PRON-']

    # Remove punctuation: no_punc
    no_punc = ' '.join(re.sub(r'[^\w\s]', '', t) for t in lemmas)

    # add list comprehension to remove digits

    return no_punc


# define stopwords variable by applying lemma_nopunc function to user-defined list of stop words: my_stopwords
my_stopwords = lemma_nopunc([
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
    'briefing',
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
    'evening',
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
    "let's",
    'like',
    'me',
    'more',
    'morning',
    'most',
    'movie',
    'movies',
    'my',
    'myself',
    'need',
    'know',
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
]).split()


def date_feats(df, date_col):
    """
    This function generates new date features from an existing date Series in a pandas DataFrame.
    It returns the pandas DataFrame passed in by the user with new date features.

    @param df: pandas DataFrame that contains original date feature
    @param date_col: date feature to be used to generate new derivative date features
    @return: pandas DataFrame with newly generated date features ^^^
    """
    # convert df[date_col] to datetime data type: df[date_col]
    df[date_col] = pd.to_datetime(df[date_col])
    # scrape month from df[date_col]: df['year']
    df['year'] = df[date_col].dt.year
    # scrape month from df[date_col]: df['month']
    df['month'] = df[date_col].dt.month
    # scrape day from df[date_col]: df['day']
    df['day'] = df[date_col].dt.day
    # scrape dayofweek from df[date_col]: df['dayofweek']
    df['dayofweek'] = df[date_col].dt.dayofweek
    # scrape hour from df[date_col]: df['hour']
    df['hour'] = df[date_col].dt.hour
    # reset index & drop previous index column
    df = df.reset_index().drop('index', axis=1)

    print('Generated Date Features & reset index!')
    print()
    print(df.columns)
    print()

    return df


def tb_sentiment(text):
    """
    This function generates sentiment labels from an existing text Series.
    It returns the polarity and subjectivity scores generated by TextBlob

    @param text: text to be scored by TextBlob sentiment analysis model
    @return: polarity and subjectivity scores generated by TextBlob model
    """
    return TextBlob(text).sentiment


def sentiment_analyzer(df, text_feature):
    """
    This function applies sentiment labels row-wise using a text-based column feature
    in a Pandas DataFrame. It joins the sentiment scores generated with the source DataFrame

    @param df: pandas DataFrame that holds text feature to be used for modeling
    @param text_feature: pandas Series text feature to be used for sentiment modeling
    @return: original pandas DataFrame joined to model sentiment scores
    """
    # create a new DataFrame filled with scores: tb_sentiment_df
    tb_sent_scores = [tb_sentiment(text=row) for row in df[text_feature]]
    tb_sentiment_df = pd.DataFrame(tb_sent_scores)

    # join sentiment scores to original article_df DataFrame
    df = df.merge(tb_sentiment_df, left_index=True, right_index=True)

    print('Generated TextBlob Sentiment Scores!')
    print()
    print(df[df['polarity'] == df['polarity'].max()][text_feature].values)
    print()

    return df


def sentiment_label(df, col_for_label, label_col, text_feature, contains_term):
    """
    This function generates the labels for sentiment analysis: ['positive','neutral','negative']
    by utilizing a np.where function:
    If the text_feature contains a user-selected term then the function will return 'negative',
    otherwise it will follow the Vader sentiment scoring logic mentioned here:
        https://github.com/cjhutto/vaderSentiment#about-the-scoring
    This function returns a DataFrame containing the new label column.

    @param df: pandas DataFrame that contains Series to be used to generate labels
    @param col_for_label: pandas Series to be used to generate labels
    @param label_col: name of pandas Series that will hold newly generated labels
    @param text_feature: name of pandas Series containing the text feature to search for user-selected term
    @param contains_term: word that user will define to search for in text feature
    @return: pandas DataFrame with newly generated labels ^^^
    """
    # if text_feature contains user-defined term, return 'negative'
    df[label_col] = np.where(
        df[text_feature].str.contains(
            contains_term,
            case=False
        ),
        'negative',

        # if df[col_for_label] >= 0.05, return 'positive'
        np.where(
            df[col_for_label] >= 0.05,
            'positive',

            # otherwise if df[col_for_label] is greater than -0.05 & less than 0.05, return 'neutral'
            np.where(
                (df[col_for_label] > -0.05) & (df[col_for_label] < 0.05),
                'neutral',

                # otherwise if df[col_for_label] <= -0.05, return 'negative'
                np.where(
                    df[col_for_label] <= -0.05,
                    'negative',

                    # else return np.nan
                    np.nan
                )
            )
        )
    )

    print('Computed labels for Modeling')
    print()
    print(df[label_col].unique())
    print()
    print(df[label_col].value_counts())
    print()

    return df


def char_count(df, text, new_pd_series):
    """
    This function counts the number of characters per row of text in a DataFrame series.
    The function generates a new pandas Series containing the number of characters per row.
    It returns a DataFrame containing the new column

    @param df: pandas DataFrame containing text Series
    @param text: pandas Series to count characters
    @param new_pd_series: name of new pandas Series containing count results
    @return: pandas DataFrame with new Series ^^^
    """
    df[new_pd_series] = df[text].apply(len)

    print(df[new_pd_series].head())
    print()

    return df


def apply_func(df, pd_series, new_pd_series, func):
    """
    Apply a function to a pandas series (row-wise) and return the resulting DataFrame with the new pandas series,
    containing the applied result.

    @param df: pandas DataFrame that contains Series to apply function
    @param pd_series: pandas Series to apply function on
    @param new_pd_series: new pandas Series that will contain function results
    @param func: function to apply over pnadas Series' rows
    @return: pandas DataFrame with new Series ^^^
    """
    # apply function to pandas Series row-wise fashion
    df[new_pd_series] = df[pd_series].apply(func)

    # print first 5 rows of new Series
    print(df[new_pd_series].head())
    print()
    print(df.columns)
    print()

    return df


def drop_high_corr(df):
    """
    Drop highly-correlated features (any feature that has > .79 correlation with another feature) from a DataFrame

    @param df: DataFrame that contains features to potentially drop
    @return: DataFrame that does not contains highly-correlated features to potentially drop
    """
    # Calculate the correlation matrix and take the absolute value: corr_matrix
    corr_matrix = df.corr().abs()

    # Create a True/False mask and apply it: tri_df
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    # List column names of highly correlated features (r > 0.79): to_drop
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.79)]

    # Drop the features in the to_drop list: reduced_df
    reduced_df = df.drop(to_drop, axis=1)

    print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))
    print()

    return reduced_df


def get_dummy_cats(df, feat):
    # convert feature to dummy columns using pandas get_dummies()
    dummy_df = pd.get_dummies(df[feat])

    # print new columns
    print(dummy_df.columns)
    print()

    # join to original pandas DataFrame
    df = df.join(dummy_df)

    # drop one column to avoid the dummy trap
    df = df.drop('Cory Booker', axis=1)

    return df

def get_vocab_size(text_list):
    """
    This function will accept a list of text feature as input and return a count of unique words in the vocabulary.

    @param text_list: list of text feature
    @return: count of unique words in the vocabulary list
    """
    # Transform the list of sentences into a list of words
    all_words = ' '.join(text_list).split(' ')

    # Get number of unique words
    unique_words = list(set(all_words))

    # # Dictionary of indexes as keys and words as values
    # index_to_word = {i: wd for i, wd in enumerate(sorted(unique_words))}

    # Dictionary of words as keys and indexes as values
    word_to_index = {wd: i for i, wd in enumerate(sorted(unique_words))}

    # # print dictionaries
    # print(word_to_index)
    # print()
    # print(index_to_word)
    # print()

    # return length of unique words in vocabulary
    return len(unique_words), word_to_index