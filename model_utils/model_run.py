"""
*******************************************************************************************************************
model_utils.model_run

This module contains customized utilities for making Sentiment Analysis predictions:
    - ml_predict_sentiment (make sentiment predictions using stacked model pipeline)
    - rnn_predict_sentiment (make sentiment predictions using recurrent neural network)

Created on 12/31/19 by William Scardino
Last updated: 2/24/20
*******************************************************************************************************************
"""
import numpy as np
import pandas as pd
import pickle
import datetime as dt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import load_model


def ml_predict_sentiment(
        source_df,
        model_df,
        text_feature,
        num_features,
        label,
        candidate_list,
        text_model_pkl,
        num_model_pkl,
        stack_model_pkl):
    """
    Load text model, numeric model and stacked model from pickle files and make predictions with Stacked model.
    Return a pandas DataFrame with predictions joined to the original source DataFrame.

    @param source_df: Source pandas DataFrame from which features were derived from
    @param model_df: pandas DataFrame containing features & labels for modeling
    @param text_feature: text feature used for modeling
    @param num_features: numeric features used for modeling
    @param label: target variable used for validating predictions
    @param text_model_pkl: path of Text Model pickle file
    @param candidate_list: list of candidates to filter for results
    @param num_model_pkl: path of Numeric Model pickle file
    @param stack_model_pkl: path of Stacked Model pickle file
    @return: pandas DataFrame containing the original source data, model features, labels and predictions
    """
    # define feature set: X
    X = model_df.drop(label, axis=1)
    # define label: y
    y = model_df[label].map({'positive': 1, 'neutral': 0, 'negative': -1})

    # load Text Model
    with open(text_model_pkl, 'rb') as model_file:
        text_model = pickle.load(model_file)

    print('Loaded Text model!')
    print(text_model)
    print()

    # fit model to text data
    text_model.fit(X[text_feature], y)

    # make model predictions on text data
    X['text_pred'] = text_model.predict(X[text_feature])

    # load Model with Numeric features
    with open(num_model_pkl, 'rb') as model_file:
        num_model = pickle.load(model_file)

    print('Loaded Text model!')
    print(num_model)
    print()

    # fit model to numeric data
    num_model.fit(X[num_features], y)

    # make predictions using numeric features
    X['num_pred'] = num_model.predict(X[num_features])

    # load Stacked Model
    with open(stack_model_pkl, 'rb') as model_file:
        stacked_model = pickle.load(model_file)

    print('Loaded Stacked model!')
    print(stacked_model)
    print()

    # fit stacked model to text and numeric predictions
    stacked_model.fit(X[['text_pred', 'num_pred']], y)

    # Make stacking predictions
    X['predictions'] = stacked_model.predict(X[['text_pred', 'num_pred']])

    # join predictions to input pandas DataFrame
    predictions_df = source_df.join(X['predictions']).join(model_df, lsuffix='_SOURCE', rsuffix='_FEAT')

    # print shape of new pandas DataFrame
    print('Predictions DataFrame Shape: {}'.format(predictions_df.shape))
    print()

    # print columns of new pandas DataFrame
    print('Predictions DataFrame columns: {}'.format(predictions_df.columns))
    print()

    # split candidate name to retrieve candidate's last name:
    predictions_df['candidate_last_name'] = predictions_df['candidate'].str.split(' ', expand=True)[1].str.lower()

    # create a flag to filter results according to 2 rules
    predictions_df['flag'] = np.where(
        # rule 1: filter out articles that don't contain the candidate's last name
        predictions_df['web_url'].str.contains(
            predictions_df['candidate_last_name'].values[0],
            case=False),
        1,
        # rule 2: filter out articles that contain candidate's who are no longer running for President
        np.where(
            predictions_df['candidate_last_name'].isin(candidate_list),
            2,
            0
        )
    )

    # filter results by the flag: filtered_df
    filtered_df = predictions_df[predictions_df['flag'] > 0]
    print(filtered_df['candidate'].value_counts())
    print()

    # group average sentiment by candidate
    grouped_df = filtered_df.groupby('candidate')['predictions'].mean()\
        .reset_index()\
        .sort_values(by='predictions', ascending=False)

    # print candidate average sentiment
    print(grouped_df)
    print()

    # write final pandas DataFrame containing predictions to .csv file
    predictions_df.to_csv('Stacked_nyt_sentiment_predictions_{date:%Y.%m.%d}.csv'.format(date=dt.datetime.now()),
                          index=False)


def rnn_predict_sentiment(model_df, source_df, text_feature, max_length, label, epochs, candidate_list, model_file_name):
    # define feature set: X
    X = model_df[text_feature]
    # define label: y
    y = model_df[label].map({'positive': 1, 'neutral': 0, 'negative': -1})

    # Create and fit tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    # Prepare the data
    prep_data = tokenizer.texts_to_sequences(X)
    prep_data = pad_sequences(prep_data, maxlen=max_length)

    # Prepare the labels
    prep_labels = to_categorical(y)

    # Print the shapes
    print(prep_data.shape)
    print()
    print(prep_labels.shape)
    print()

    # load model
    model = load_model(model_file_name)

    # summarize model
    model.summary()

    # Fit the network
    model.fit(prep_data, prep_labels, epochs=epochs)

    # Use the model to predict on new data
    predicted = model.predict(X)

    # Choose the class with higher probability
    y_pred = np.argmax(predicted, axis=1)

    # join predictions to input pandas DataFrame
    predictions_df = source_df.join(pd.DataFrame(y_pred, columns='predictions')).join(model_df, lsuffix='_SOURCE', rsuffix='_FEAT')

    # print shape of new pandas DataFrame
    print('Predictions DataFrame Shape: {}'.format(predictions_df.shape))
    print()

    # print columns of new pandas DataFrame
    print('Predictions DataFrame columns: {}'.format(predictions_df.columns))
    print()

    # split candidate name to retrieve candidate's last name:
    predictions_df['candidate_last_name'] = predictions_df['candidate'].str.split(' ', expand=True)[1].str.lower()

    # create a flag to filter results according to 2 rules
    predictions_df['flag'] = np.where(
        # rule 1: filter out articles that don't contain the candidate's last name
        predictions_df['web_url'].str.contains(
            predictions_df['candidate_last_name'].values[0],
            case=False),
        1,
        # rule 2: filter out articles that contain candidate's who are no longer running for President
        np.where(
            predictions_df['candidate_last_name'].isin(candidate_list),
            2,
            0
        )
    )

    # filter results by the flag: filtered_df
    filtered_df = predictions_df[predictions_df['flag'] > 0]
    print(filtered_df['candidate'].value_counts())
    print()

    # group average sentiment by candidate
    grouped_df = filtered_df.groupby('candidate')['predictions'].mean() \
        .reset_index() \
        .sort_values(by='predictions', ascending=False)

    # print candidate average sentiment
    print(grouped_df)
    print()

    # write final pandas DataFrame containing predictions to .csv file
    predictions_df.to_csv('RNN_nyt_sentiment_predictions_{date:%Y.%m.%d}.csv'.format(date=dt.datetime.now()),
                          index=False)
