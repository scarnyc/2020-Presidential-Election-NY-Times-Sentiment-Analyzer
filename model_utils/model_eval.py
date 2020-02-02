"""
*******************************************************************************************************************
model_utils.model_eval

This package contains customized utilities for training Sentiment Analysis models:
    - split_df (splits DataFrame into KFold DataFrames)
    - text_model_metrics (iterate over a list of models, fitting them and getting evaluation metrics on text data)
    - num_model_metrics(iterate over a list of models, fitting them and getting evaluation metrics on numeric data)
    - stacked_model_metrics (fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics)

created: 1/5/20
last updated: 1/28/20
*******************************************************************************************************************
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import pickle


def split_df(df, label_col):
    """
    Splits a pandas DataFrame into a list of DataFrames based on candidate text.
    @param df:
    @param label_col:
    @return:
    """

    df1 = df.sample(frac=0.2, random_state=1)
    df2 = df.sample(frac=0.2, random_state=2)
    df3 = df.sample(frac=0.2, random_state=3)
    df4 = df.sample(frac=0.2, random_state=4)
    df5 = df.sample(frac=0.2, random_state=5)

    for df in [
        df1,
        df2,
        df3,
        df4,
        df5
    ]:
        print('DataFrame shape: {}'.format(df.shape))
        print()
        print('DataFrame head: {}'.format(df.head()))
        print()
        print('DataFrame labels; {}'.format(df[label_col].value_counts()))
        print()

    return [df1, df2, df3, df4, df5]


def text_model_metrics(models, vectorizers, df, text_feature, label):
    """
    Iterate over a list of models, fitting them and getting evaluation metrics on text data.
    """
    # split DataFrame into 5-Folds
    kfold_list = split_df(
        df=df,
        label_col=label
    )
    print('Split DataFrames into 5-Fold datasets!')
    print()

    # Start Text model training
    print('Text Model Results!')
    print()

    # iterate over model list
    for model in models:
        print(model)
        print()

        # iterate over vectorizer list
        for vectorizer in vectorizers:
            print(vectorizer)
            print()

            # instantiate model training pipeline
            pipe = Pipeline([('vectorizer', vectorizer),
                             ('scaler', StandardScaler(with_mean=False)),
                             ('clf', model)
                             ])
            print(pipe)
            print()

            # instantiate lists for loop
            acc = []
            f1 = []

            # iterate over list of KFold DataFrames
            for fold in kfold_list:
                # define feature set: X
                X = fold[text_feature]
                # define label: y
                y = fold[label]
                # split each fold into training & test set:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
                print('Training set shape: {}'.format(X_train.shape))
                print()
                print('Test set shape: {}'.format(X_test.shape))
                print()

                # Fit the classifier
                pipe.fit(X_train, y_train)

                # Predict test set labels & probabilities
                y_pred = pipe.predict(X_test)

                # see if model is overfitting
                print('Training Set Accuracy')
                print(pipe.score(X_train, y_train))
                print()
                print('Test Set Accuracy')
                print(pipe.score(X_test, y_test))
                print()

                # append accuracy score to list: acc
                acc.append(pipe.score(X_test, y_test))

                # append accuracy score to list: f1
                f1.append(f1_score(y_test, y_pred, average='micro'))

                # Compute and print the confusion matrix and classification report
                print('Confusion matrix')
                print(confusion_matrix(y_test, y_pred))
                print()
                print('Classification report')
                print(classification_report(y_test, y_pred))
                print()
                print()

            print('5-fold cross-validated Accuracy: {}'.format(np.mean(acc)))
            print()
            print('5-fold cross-validated F1 score: {}'.format(np.mean(f1)))
            print()


def num_model_metrics(models, df, label, num_features):
    """
    Iterate over a list of models, fitting them and getting evaluation metrics on numeric data.
    @param models:
    @param df:
    @param label:
    @param num_features:
    @return:
    """
    # split DataFrame into 5-Folds
    kfold_list = split_df(
        df=df,
        label_col=label
    )
    print('Split DataFrames into 5-Fold datasets!')
    print()

    # Start Numeric model training
    print('Numeric Model Results!')
    print()

    # iterate over model list
    for model in models:
        print(model)
        print()

        # instantiate model training pipeline
        pipe = Pipeline([
                         ('scaler', MinMaxScaler()),
                         ('clf', model)
                         ])
        print(pipe)
        print()

        # instantiate lists for loop
        acc = []
        f1 = []

        # iterate over list of KFold DataFrames
        for fold in kfold_list:
            # define feature set: X
            X = fold[num_features]
            # define label: y
            y = fold[label]
            # split each fold into training & test set:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
            print('Training set shape: {}'.format(X_train.shape))
            print()
            print('Test set shape: {}'.format(X_test.shape))
            print()

            # Fit the classifier
            pipe.fit(X_train, y_train)

            # Predict test set labels & probabilities
            y_pred = pipe.predict(X_test)

            # see if model is overfitting
            print('Training Set Accuracy')
            print(pipe.score(X_train, y_train))
            print()
            print('Test Set Accuracy')
            print(pipe.score(X_test, y_test))
            print()

            # append accuracy score to list: acc
            acc.append(pipe.score(X_test, y_test))

            # append accuracy score to list: f1
            f1.append(f1_score(y_test, y_pred, average='micro'))

            # Compute and print the confusion matrix and classification report
            print('Confusion matrix')
            print(confusion_matrix(y_test, y_pred))
            print()
            print('Classification report')
            print(classification_report(y_test, y_pred))
            print()
            print()

        print('5-fold cross-validated Accuracy: {}'.format(np.mean(acc)))
        print()
        print('5-fold cross-validated F1 score: {}'.format(np.mean(f1)))
        print()


def text_random_hyper(df, text_feature, label, model, vectorizer, n_iters, cfv):

    # define feature set: X
    X = df[text_feature]
    # define label: y
    y = df[label]
    # split each fold into training & test set:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    print('Training set shape: {}'.format(X_train.shape))
    print()
    print('Test set shape: {}'.format(X_test.shape))
    print()

    # instantiate model training pipeline
    pipe = Pipeline([('vectorizer', vectorizer),
                     ('scaler', StandardScaler(with_mean=False)),
                     ('clf', model)
                     ])
    print(pipe)
    print()

    # Create the parameter grid
    param_grid = {
        'clf__colsample_bytree': [0.3, 0.7],
        'clf__n_estimators': [100, 500, 1000],
        'clf__max_depth': [3, 6, 10, 20],
        'clf__learning_rate': np.linspace(.1, 2, 150),
        'clf__min_samples_leaf': list(range(20, 65))
    }

    # Create a random search object
    random_model = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=n_iters,
        scoring='accuracy',
        n_jobs=4,
        cv=cfv,
        refit=True,
        return_train_score=True,
        verbose=1
    )

    # Fit to the training data
    random_model.fit(X_train, y_train)

    # Print the values used for both Parameters & Score
    print("Best random Parameters: ", random_model.best_params_)
    print()
    print("Best random Score: ", random_model.best_score_)
    print()

    return random_model.best_estimator_

# def stacked_model_metrics(
#         train1_df,
#         train2_df,
#         test_df,
#         y_test,
#         label_col,
#         text_model,
#         text_feature,
#         text_prediction_col,
#         n_gram_range,
#         k,
#         stopwords,
#         text_model_pkl,
#         num_model,
#         num_train1_features,
#         num_features,
#         num_prediction_col,
#         num_model_pkl,
#         stacked_model,
#         stacked_features,
#         stacked_model_pkl
# ):
#     """
#     Fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics.
#
#     """
#     # Text Model Pipeline
#     print('Stacked Model Training!')
#     print('Text Model!')
#     text_pipe = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=n_gram_range, stop_words=stopwords)),
#                           ('feature_select', SelectKBest(chi2, k=k)),
#                           ('scaler', StandardScaler(with_mean=False)),
#                           ('text_clf', text_model)
#                           ])
#
#     print(text_pipe)
#     print()
#
#     # Fit the classifier
#     text_pipe.fit(train1_df[text_feature], train1_df[label_col])
#
#     # Predict test set labels
#     train2_df[text_prediction_col] = text_pipe.predict(train2_df[text_feature])
#     test_df[text_prediction_col] = text_pipe.predict(test_df[text_feature])
#
#     # save text model for later
#     with open(text_model_pkl, 'wb') as model_file:
#         pickle.dump(text_pipe, model_file)
#
#     # Numeric Model: xgb_clf
#     num_pipe = Pipeline([
#         ('scaler', MinMaxScaler()),
#         ('num_clf', num_model)
#     ])
#
#     # Numeric Model Pipeline
#     print('Numeric Model!')
#     print(num_pipe)
#     print()
#
#     # Fit the classifier
#     num_pipe.fit(num_train1_features, train1_df[label_col])
#
#     # Predict test set labels
#     train2_df[num_prediction_col] = num_pipe.predict(train2_df[num_features])
#     test_df[num_prediction_col] = num_pipe.predict(test_df[num_features])
#
#     # save model for later
#     with open(num_model_pkl, 'wb') as model_file:
#         pickle.dump(num_pipe, model_file)
#
#     # Stacked Model
#     print('Stacked Model!')
#     print(stacked_model)
#
#     # Train 2nd level model on the Part 2 data
#     stacked_model.fit(train2_df[[text_prediction_col, num_prediction_col]], train2_df[label_col])
#
#     # Make stacking predictions on the test data
#     test_df['stacking'] = stacked_model.predict(test_df[[text_prediction_col, num_prediction_col]])
#
#     # Look at the model coefficients
#     print('LogisticRegression Coefs: {}'.format(stacked_model.coef_))
#     print()
#     print('Training set Accuracy: {}'.format(stacked_model.score(train2_df[stacked_features], train2_df[label_col])))
#     print()
#     print('Test set Accuracy: {}'.format(stacked_model.score(test_df[stacked_features], y_test)))
#     print()
#
#     # Compute and print the confusion matrix and classification report
#     print('Confusion matrix')
#     print(confusion_matrix(y_test, test_df['stacking']))
#     print()
#     print('Classification report')
#     print(classification_report(y_test, test_df['stacking']))
#     print()
#
#     # save model for later
#     with open(stacked_model_pkl, 'wb') as model_file:
#         pickle.dump(stacked_model, model_file)
#
