"""
*******************************************************************************************************************
model_utils.model_eval

This package contains customized utilities for training Sentiment Analysis models:
    - text_model_metrics (iterate over a list of models, fitting them and getting evaluation metrics on text data)
    - num_model_metrics(iterate over a list of models, fitting them and getting evaluation metrics on numeric data)
    - stacked_model_metrics (fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics)

created: 1/5/20
*******************************************************************************************************************
"""
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import pickle


def text_model_metrics(models, X_train, X_test, y_train, y_test, text_feature, n_gram_range, k, stopwords):
    """
    Iterate over a list of models, fitting them and getting evaluation metrics on text data.

    @param models:
    @param X_train:
    @param X_test:
    @param y_train:
    @param y_test:
    @param text_feature:
    @param n_gram_range:
    @param k:
    @param stopwords:
    @return:
    """
    print('Text Model Results!')
    for model in models:
        print(model)
        print()
        pipe = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=n_gram_range, stop_words=stopwords)),
                         ('feature_select', SelectKBest(chi2, k=k)),
                         ('scaler', StandardScaler(with_mean=False)),
                         ('clf', model)
                         ])

        print(pipe)
        print()

        # Fit the classifier
        pipe.fit(X_train[text_feature], y_train)

        # Predict test set labels & probabilities
        y_pred = pipe.predict(X_test[text_feature])

        # Compute and print the confusion matrix and classification report
        print('Confusion matrix')
        print(confusion_matrix(y_test, y_pred))
        print()
        print('Classification report')
        print(classification_report(y_test, y_pred))
        print()
        print()
        print()


def num_model_metrics(models, X_train, X_test, y_train, y_test, num_features):
    """
    Iterate over a list of models, fitting them and getting evaluation metrics on numeric data.

    @param models:
    @param X_train:
    @param X_test:
    @param y_train:
    @param y_test:
    @param num_features:
    @return:
    """
    print('Numeric Model Results!')
    for model in models:
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', model)
        ])

        print(pipe)
        print()

        # Fit the classifier
        pipe.fit(X_train[num_features], y_train)

        # Predict test set labels & probabilities
        y_pred = pipe.predict(X_test[num_features])

        # Compute and print the confusion matrix and classification report
        print('Confusion matrix')
        print(confusion_matrix(y_test, y_pred))
        print()
        print('Classification report')
        print(classification_report(y_test, y_pred))
        print()
        print()
        print()


def stacked_model_metrics(
        train1_df,
        train2_df,
        test_df,
        y_test,
        label_col,
        text_model,
        text_feature,
        text_prediction_col,
        n_gram_range,
        k,
        stopwords,
        text_model_pkl,
        num_model,
        num_train1_features,
        num_features,
        num_prediction_col,
        num_model_pkl,
        stacked_model,
        stacked_features,
        stacked_model_pkl
):
    """
    Fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics.

    """
    # Text Model Pipeline
    print('Stacked Model Training!')
    print('Text Model!')
    text_pipe = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=n_gram_range, stop_words=stopwords)),
                          ('feature_select', SelectKBest(chi2, k=k)),
                          ('scaler', StandardScaler(with_mean=False)),
                          ('text_clf', text_model)
                          ])

    print(text_pipe)
    print()

    # Fit the classifier
    text_pipe.fit(train1_df[text_feature], train1_df[label_col])

    # Predict test set labels
    train2_df[text_prediction_col] = text_pipe.predict(train2_df[text_feature])
    test_df[text_prediction_col] = text_pipe.predict(test_df[text_feature])

    # save text model for later
    with open(text_model_pkl, 'wb') as model_file:
        pickle.dump(text_pipe, model_file)

    # Numeric Model: xgb_clf
    num_pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('num_clf', num_model)
    ])

    # Numeric Model Pipeline
    print('Numeric Model!')
    print(num_pipe)
    print()

    # Fit the classifier
    num_pipe.fit(num_train1_features, train1_df[label_col])

    # Predict test set labels
    train2_df[num_prediction_col] = num_pipe.predict(train2_df[num_features])
    test_df[num_prediction_col] = num_pipe.predict(test_df[num_features])

    # save model for later
    with open(num_model_pkl, 'wb') as model_file:
        pickle.dump(num_pipe, model_file)

    # Stacked Model
    print('Stacked Model!')
    print(stacked_model)

    # Train 2nd level model on the Part 2 data
    stacked_model.fit(train2_df[[text_prediction_col, num_prediction_col]], train2_df[label_col])

    # Make stacking predictions on the test data
    test_df['stacking'] = stacked_model.predict(test_df[[text_prediction_col, num_prediction_col]])

    # Look at the model coefficients
    print('LogisticRegression Coefs: {}'.format(stacked_model.coef_))
    print()
    print('Training set Accuracy: {}'.format(stacked_model.score(train2_df[stacked_features], train2_df[label_col])))
    print()
    print('Test set Accuracy: {}'.format(stacked_model.score(test_df[stacked_features], y_test)))
    print()

    # Compute and print the confusion matrix and classification report
    print('Confusion matrix')
    print(confusion_matrix(y_test, test_df['stacking']))
    print()
    print('Classification report')
    print(classification_report(y_test, test_df['stacking']))
    print()

    # save model for later
    with open(stacked_model_pkl, 'wb') as model_file:
        pickle.dump(stacked_model, model_file)

