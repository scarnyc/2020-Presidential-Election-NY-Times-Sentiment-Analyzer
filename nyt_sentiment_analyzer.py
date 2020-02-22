#!/usr/bin/env python
# coding: utf-8
"""
**********************************************************************************************
Sentiment Analysis on 2020 U.S. Presidential Candidates N.Y. Times Articles (.py script)

Created on 12/31/19 by William Scardino
Last updated: 2/21/20
**********************************************************************************************
"""
# Import custom packages
from core_utils.dataframe import union_csv
from custom_utils.clean_dataframe import preprocess_df, filter_dataframe
from model_utils.feature_eng import (date_feats, my_stopwords, get_vocab_size,
                                     char_count, sentiment_label, lemma_nopunc,
                                     apply_func, drop_high_corr, sentiment_analyzer)
from viz_utils.viz import (corr_heatmap, plot_word_freq, two_dim_tf_viz,
                           time_series_line_viz)
from model_utils.model_eval import (model_training_metrics, num_feature_importance,
                                    text_feature_importance, neural_net_train_metrics,
                                    model_random_hyper_tune, stacked_model_metrics)
from model_utils.model_run import predict_sentiment

# Import data science packages
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def sentiment_analysis_pipe(directory):
    """
    This script performs EDA on data & trains a stacked machine learning model to perform sentiment analysis of 2020 U.S. Presidential Candidates,
    using the N.Y. Times Article Search API.

     Please note the following functions that are used in the Sentiment Analysis Model Pipeline:

    - union_csv (reads .csv files containing Article abstract descriptions of N.Y. Times articles into a single pandas DataFrame.
        Please note that the .csv files were created by calling the nyt_api.main module)
    - preprocess_df (pre-processes aforementioned DataFrame by dropping null values and duplicates,
        and also concatenating columns for text feature engineering)
    - date_feats (generates year, month, day and hour date features from existing date feature)
    - sentiment_analyzer (generates TextBlob Sentiment scores to be used as labels row by row over the entire pandas DataFrame)
    - sentiment_label (groups raw TextBlob Sentiment scores into 3 labels: ['positive','negative','neutral'])
    - char_count (counts length of text feature)
    - apply_func (applies the lemma_nopunc function row by row over the pandas DataFrame, effectively lemmatizing the text feature)
    - corr_heatmap (plots a HeatMap of feature correlations for EDA)
    - drop_high_corr (automatically drops highly-correlated features from a pandas DataFrame)
    - plot_word_freq (plots the top N features from a BOW model, for EDA)
    - two_dim_tf_viz (plots a 2D scatter plot of TfIdf scores of text features, colored by sentiment label for EDA)
    - time_series_line_viz (plots a line plot of Sentiment over time for a given pandas DataFrame for EDA)
    - model_training_metrics (print model evaluation metrics, iterating over a list of models that are passed into a list,
        allowing for the comparison of different models and their results to guide choice for stacking purposes)
    - model_random_hyper_tune (perform hyper-parameter tuning on any given model that is chosen from evaluation,
        by utilizing a Random Search)
    - text_feature_importance (print the top 25 features from the text model by their TfIdf weights
    - num_feature_importance (print the most important features for tree-based models with numeric features)
    - stacked_model_metrics (finally, fit both the numeric & text models that were previously tuned performing cross-validation,
        and adding a second-layer LogisticRegression stacked model that will use the predictions from the other two models as features,
        for the final predictions)
    - predict_sentiment (make predictions on N.Y. Times article data using stacked model)
    - get_vocab_size (retreive a count of unique words in a vocabulary)
    - neural_net_train_metrics (build, compile and train a recurrent neural network and get evaluation metrics)

    @param directory: directory containing .csv files scraped from the N.Y. Times Article Search API
    """

    # read .csv files & union them into a single DataFrame: trump_df
    article_df = union_csv(
        csv_path=directory,
        glob_pattern='*.csv'
    )

    # return preprocessed DataFrame with 'headline_main','web_url','text','word_count','pub_date' columns
    article_df = preprocess_df(
        df=article_df,
        new_col='text',
        con_col1='headline_main',
        con_col2='abstract',
        subset_list=['abstract'],
        filter_col_list=['headline_main', 'web_url', 'text', 'word_count', 'pub_date', 'candidate']
    )

    # generate date_features for article_df
    article_df = date_feats(df=article_df, date_col='pub_date')

    # filter out articles written prior to U.S. Presidential election campaigns (ie. 2019)
    article_df = filter_dataframe(df=article_df, col_to_filter='year', value_to_filter=2019)

    # generate TextBlob sentiment and subjectivity scores for each row of 'text' in article_df: tb_sent_scores
    article_df = sentiment_analyzer(df=article_df, text_feature='text')

    # count number of characters: article_df['char_count']
    article_df = char_count(
        df=article_df,
        text='text',
        new_pd_series='char_count'
    )

    # lemmatize words & remove punctuation: article_df['text_feat']
    article_df = apply_func(
        df=article_df,
        pd_series='text',
        new_pd_series='text_feat',
        func=lemma_nopunc
    )

    # compute the labels for modelling: article_df['sentiment_label']
    article_df = sentiment_label(
        df=article_df,
        col_for_label='polarity',
        label_col='sentiment_label',
        text_feature='text_feat',
        contains_term='impeach'
    )

    # return column subsets of article_df for modeling: model_df
    model_df = article_df[
        ['text_feat', 'year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity', 'char_count',
         'sentiment_label']
    ]

    # plot heatmap of feature correlations
    corr_heatmap(df=article_df,
                 features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity']
                 )

    # automatically remove highly correlated features
    model_df = drop_high_corr(df=model_df)

    # plot word frequencies
    plot_word_freq(
        pd_series=article_df[article_df['sentiment_label'] == 'positive']['text'],
        plot_title='BOW FREQUENCY',
        stopwords=my_stopwords,
        n=30
    )

    # plot tfidf Scatter plot
    two_dim_tf_viz(
        df=article_df,
        pd_series='text',
        stopwords=my_stopwords,
        pd_color_series='sentiment_label',
        max_features=1000,
        plot_title='N.Y. Times Article Sentiment Clusters'
    )

    # plot Time Series Line plot
    time_series_line_viz(
        df=article_df[article_df['text'].str.contains('Trump', case=False)],
        date_index='pub_date',
        pd_series='polarity',
        plot_title='N.Y. Times Articles Avg. Daily Sentiment'
    )

    # instantiate list of models: models
    text_models = [
        Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(MultinomialNB()))
                  ]),
        Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('scaler', StandardScaler(with_mean=False)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(MultinomialNB()))
                  ]),
        Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(SVC(probability=True, random_state=42, class_weight='balanced')))
                  ]),
        Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('scaler', StandardScaler(with_mean=False)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(SVC(probability=True, random_state=42, class_weight='balanced')))
                  ]),
        Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42,
                                                                     n_jobs=4, class_weight='balanced')))
                  ]),
        Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('scaler', StandardScaler(with_mean=False)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42,
                                                                     n_jobs=4, class_weight='balanced')))
                  ]),
        Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                 ('dim_red', SelectKBest(chi2, k=300)),
                 ('clf', OneVsRestClassifier(XGBClassifier(n_jobs=4, random_state=42)))
                 ]),
        Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('scaler', StandardScaler(with_mean=False)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(XGBClassifier(n_jobs=4, random_state=42)))
                   ]),
        Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf',OneVsRestClassifier(LogisticRegression(max_iter=5000, solver='liblinear',
                                                                random_state=42, class_weight='balanced')))
                   ]),
        Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stopwords)),
                  ('scaler', StandardScaler(with_mean=False)),
                  ('dim_red', SelectKBest(chi2, k=300)),
                  ('clf', OneVsRestClassifier(LogisticRegression(max_iter=5000, solver='liblinear',
                                                                 random_state=42, class_weight='balanced')))
                   ])
            ]
    print('Instantiated list of text models!')
    print()

    # print out text model metrics
    model_training_metrics(
        df=model_df,
        models=text_models,
        features='text_feat',
        label='sentiment_label'
    )

    # instantiate list of models: models
    num_models = [
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(SVC(probability=True, random_state=42, class_weight='balanced')))
        ]),
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42,
                                                               n_jobs=4, class_weight='balanced')))
        ]),
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(XGBClassifier(n_jobs=4, random_state=42)))
        ]),
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
                                                           random_state=42, class_weight='balanced')))
        ])
    ]
    print('Instantiated list of models with numeric features!')
    print()

    # print out metrics for models with numeric features
    model_training_metrics(
        df=model_df,
        models=num_models,
        features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
        label='sentiment_label'
    )

    # tune hyper parameters for model with numeric feature inputs: num_pipe, cv_results_df, model_params
    text_pipe, text_cv_results_df = model_random_hyper_tune(
        df=model_df,
        model=Pipeline([('vectorizer', CountVectorizer(stop_words=my_stopwords)),
                        ('scaler', StandardScaler(with_mean=False)),
                        ('dim_red', SelectKBest(chi2)),
                        ('clf', OneVsRestClassifier(XGBClassifier(random_state=42)))
                        ]),
        param_grid={
            'vectorizer__ngram_range': [(1, 3), (2, 3)],
            'dim_red__k': [100, 200, 300],
            'clf__estimator__booster': ['gbtree', 'gblinear', 'dart'],
            'clf__estimator__colsample_bytree': [0.3, 0.7],
            'clf__estimator__n_estimators': [100, 200, 300],
            'clf__estimator__max_depth': [3, 6, 10, 20],
            'clf__estimator__learning_rate': np.linspace(.1, 2, 150),
            'clf__estimator__min_samples_leaf': list(range(20, 65))
        },
        features='text_feat',
        label='sentiment_label',
        n_iters=25,
        n_folds=5,
        model_file_path="./models/text_pipe_xgb.pkl"
    )

    # get feature importances from TFIDF scores: tfidf_df
    text_feature_importance(
        df=model_df[model_df['sentiment_label'] == 'negative'],
        text_feature='text_feat',
        vectorizer=text_pipe[0]
    )

    # tune hyper parameters for model with numeric feature inputs: num_pipe, cv_results_df, model_params
    num_pipe, num_cv_results_df = model_random_hyper_tune(
        df=model_df,
        model=Pipeline([('scaler', StandardScaler()),
                        ('clf', OneVsRestClassifier(XGBClassifier(booster='gbtree', random_state=42)))
                        ]),
        param_grid={
                    'clf__estimator__colsample_bytree': [0.3, 0.7],
                    'clf__estimator__n_estimators': [100, 200, 300],
                    'clf__estimator__max_depth': [3, 6, 10, 20],
                    'clf__estimator__learning_rate': np.linspace(.1, 2, num=50),
                    'clf__estimator__min_samples_leaf': list(range(20, 60))
                    },
        features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
        label='sentiment_label',
        n_iters=25,
        n_folds=5,
        model_file_path="./models/num_pipe_xgb.pkl"
    )

    # look at most important features for text model
    num_feature_importance(df=model_df,
                           model=num_pipe[1],
                           features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count',
                                     'subjectivity']
                           )

    # print out stacked model metrics
    stacked_model_metrics(
        df=model_df,
        label='sentiment_label',
        text_model=text_pipe,
        text_feature='text_feat',
        num_model=num_pipe,
        num_features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
        stacked_model=OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
                                                             random_state=42, class_weight='balanced'))
    )

    # Make predictions using Stacked Model
    predict_sentiment(
        source_df=article_df,
        model_df=model_df,
        text_feature='text_feat',
        num_features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
        label='sentiment_label',
        text_model_pkl="./models/text_pipe_xgb.pkl",
        num_model_pkl="./models/num_pipe_xgb.pkl",
        stack_model_pkl="./models/lr_stack.pkl",
        candidate_list=['Sanders', 'Trump', 'Warren', 'Harris', 'Biden', 'Buttigieg', 'Bloomberg',
                        'Klobuchar'])

    # get the vocabulary size for the neural network: vocab_size
    vocab_size, vocab_dict = get_vocab_size(model_df['text_feat'].tolist())

    # print out metrics for neural network text model
    neural_net_train_metrics(
        df=model_df,
        text_feature='text_feat',
        max_length=model_df['char_count'].max(),
        label='sentiment_label',
        vocabulary_size=vocab_size,
        num_classes=2,
        epochs=100,
        batch_size=64,
        word2vec_dim=300,
        vocabulary_dict=vocab_dict,
        glove_file_name=r'.\glove_6B\glove.6B.300d.txt'
    )


# run the script
if __name__ == '__main__':
    sentiment_analysis_pipe(directory=r'.\data')
