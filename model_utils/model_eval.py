"""
*******************************************************************************************************************
model_utils.model_eval

This module contains customized utilities for training Sentiment Analysis models:
    - split_df (splits DataFrame into KFold DataFrames)
    - text_model_metrics (iterate over a list of models, fitting them and getting evaluation metrics on text data)
    - num_model_metrics(iterate over a list of models, fitting them and getting evaluation metrics on numeric data)
    - text_random_hyper_tune (performs hyper-parameter tuning for a text model)
    - num_random_hyper_tune (performs hyper-parameter tuning for a model with numeric feature-inputs)
    - num_feature_importance (Return a DataFrame with most important features for tree-based models with numeric features)
    - stacked_model_metrics (fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics)

created: 1/5/20
last updated: 2/13/20
*******************************************************************************************************************
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pickle


def split_df(df, label_col):
    """
    Splits a pandas DataFrame to split into a list of 5 separate DataFrames

    @param df: pandas DataFrame to split into a list of 5 separate DataFrames
    @param label_col: pandas Series that will be used as target variable for machine learning modeling
    @return: list of 5 separate DataFrames derived from original DataFrame
    """
    # split original pandas DataFrame into 5 separate DataFrames
    # each with their own random seed to effectively partition DataFrame into 5 parts
    # this will ensure that no 2 DataFrames will contain the same index
    df1 = df.sample(frac=0.2, random_state=1)
    df2 = df.sample(frac=0.2, random_state=2)
    df3 = df.sample(frac=0.2, random_state=3)
    df4 = df.sample(frac=0.2, random_state=4)
    df5 = df.sample(frac=0.2, random_state=5)

    # iterate over each of the newly created 5 DataFrames printing each DataFrame's shape,
    # first 5 rows and target value counts
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
    Iterate over a list of models, fitting them and getting evaluation metrics on text data

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
                             ('dim_red', SelectKBest(chi2, k=300)),
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
    Iterate over a list of models, fitting them and getting evaluation metrics on numeric data

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


def text_random_hyper_tune(df, text_feature, label, model, vectorizer, n_iters, n_folds):
    """
    Performs hyper-parameter tuning for a text model

    @param df:
    @param text_feature:
    @param label:
    @param model:
    @param vectorizer:
    @param n_iters:
    @param n_folds:
    @return:
    """

    # define feature set: X
    X = df[text_feature]
    # define label: y
    y = df[label]

    # instantiate model training pipeline
    pipe = Pipeline([('vectorizer', vectorizer),
                     ('scaler', StandardScaler(with_mean=False)),
                     ('dim_red', SelectKBest(chi2)),
                     ('clf', model)
                     ])
    print(pipe)
    print()

    # Create the parameter grid
    param_grid = {
        'vectorizer__ngram_range': [(1, 3), (2, 3)],
        'dim_red__k': [100, 200, 300],
        'clf__estimator__booster': ['gbtree', 'gblinear', 'dart'],
        'clf__estimator__colsample_bytree': [0.3, 0.7],
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__max_depth': [3, 6, 10, 20],
        'clf__estimator__learning_rate': np.linspace(.1, 2, 150),
        'clf__estimator__min_samples_leaf': list(range(20, 65))
    }

    # Create a grid search object
    random_model = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        scoring='accuracy',
        n_jobs=6,
        n_iter=n_iters,
        cv=n_folds,
        refit=True,
        return_train_score=True,
        verbose=1
    )

    # Fit to the training data
    random_model.fit(X, y)

    # Print the values used for both Parameters & Score
    print("Best random Parameters: ", random_model.best_params_)
    print()
    print("Best random Score: ", random_model.best_score_)
    print()

    # Read the cv_results property into a dataframe & print it out
    cv_results_df = pd.DataFrame(random_model.cv_results_).sort_values(by=["rank_test_score"])
    print(cv_results_df)
    print()

    # Extract and print the column with a dictionary of hyperparameters used
    column = cv_results_df.loc[:, ["params"]]
    print(column)
    print()

    # Extract and print the row that had the best mean test score
    best_row = cv_results_df[cv_results_df["rank_test_score"] == 1]
    print(best_row)
    print()

    # save text model for later
    with open("./models/text_pipe_xgb.pkl", 'wb') as model_file:
        pickle.dump(random_model.best_estimator_, model_file)

    return random_model.best_estimator_, cv_results_df, column


def num_random_hyper_tune(df, num_features, label, model, n_iters, n_folds):
    """
    Performs hyper-parameter tuning for a model with numeric features

    @param df:
    @param num_features:
    @param label:
    @param model:
    @param n_iters:
    @param n_folds:
    @return:
    """
    # define feature set: X
    X = df[num_features]
    # define label: y
    y = df[label]

    # instantiate model pipeline
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', model)
    ])
    print(pipe)
    print()

    # Create the parameter grid
    param_grid = {
        'clf__estimator__booster': ['gbtree', 'gblinear', 'dart'],
        'clf__estimator__colsample_bytree': [0.3, 0.7],
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__max_depth': [3, 6, 10, 20],
        'clf__estimator__learning_rate': np.linspace(.1, 2, num=50),
        'clf__estimator__min_samples_leaf': list(range(20, 60))
    }

    # Create a random search object
    random_model = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        scoring='accuracy',
        n_jobs=6,
        n_iter=n_iters,
        cv=n_folds,
        refit=True,
        return_train_score=True,
        verbose=1
    )

    # Fit to the training data
    random_model.fit(X, y)

    # Print the values used for both Parameters & Score
    print("Best random Parameters: ", random_model.best_params_)
    print()
    print("Best random Score: ", random_model.best_score_)
    print()

    # Read the cv_results property into a dataframe & print it out
    cv_results_df = pd.DataFrame(random_model.cv_results_).sort_values(by=["rank_test_score"])
    print(cv_results_df)
    print()

    # Extract and print the column with a dictionary of hyperparameters used
    column = cv_results_df.loc[:, ["params"]]
    print(column)
    print()

    # Extract and print the row that had the best mean test score
    best_row = cv_results_df[cv_results_df["rank_test_score"] == 1]
    print(best_row)
    print()

    # save model with numeric features for later
    with open("./models/num_pipe_xgb.pkl", 'wb') as model_file:
        pickle.dump(random_model.best_estimator_, model_file)

    return random_model.best_estimator_, cv_results_df, column


def text_feature_importance(df, text_feature, vectorizer):
    # https://buhrmann.github.io/tfidf-analysis.html
    # split into train & test sets
    train_df, test_df = train_test_split(df[[text_feature]], test_size=.2, random_state=42)

    # Fit the vectorizer and transform the data
    vectorizer.fit(train_df)

    # Transform test data
    tfidf_test = vectorizer.transform(test_df)

    # Create new features for the test set
    tfidf_df = pd.DataFrame(tfidf_test.toarray(),
                            columns=vectorizer.get_feature_names()).add_prefix('TFIDF_')

    # print first 5 rows of DataFrame
    print(tfidf_df.head())
    print()

    return tfidf_df


def num_feature_importance(df, model, features):
    """
    Return a DataFrame with most important features for tree-based models with numeric features

    @param df:
    @param model:
    @param features:
    @return:
    """
    # Calculate feature importance: feature_importance
    feature_importance = model.estimators_[0].feature_importances_

    # Create a list of features: feature_list
    feature_list = list(df[features])

    # Save the results inside a DataFrame using feature_list as an index: relative_importance
    relative_importance = pd.DataFrame(index=feature_list, data=feature_importance, columns=["importance"])

    # print only features with relative importance higher than 1%
    print(relative_importance[relative_importance['importance'] > 0.01].sort_values('importance', ascending=False))

    return relative_importance


def stacked_model_metrics(
        df,
        label,
        text_model,
        text_feature,
        num_model,
        num_features,
        stacked_model
):
    """
    Fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics

    @param df:
    @param label:
    @param text_model:
    @param text_feature:
    @param num_model:
    @param num_features:
    @param stacked_model:
    @return:
    """
    # split DataFrame into 5-Folds
    kfold_list = split_df(
        df=df,
        label_col=label
    )
    print('Split DataFrames into 5-Fold datasets!')
    print()
    print('Stacked Model Training!')
    print()

    # Text Model Pipeline
    print('Text Model!')
    print(text_model)
    print()

    # Numeric Model Pipeline
    print('Numeric Model!')
    print(num_model)
    print()

    # instantiate lists for loop
    acc = []
    f1 = []

    # iterate over list of KFold DataFrames
    for fold in kfold_list:
        # define feature set: X
        X = fold.drop(label, axis=1)
        # define label: y
        y = fold[label].map({'positive': 1, 'neutral': 0, 'negative': -1})

        # split each fold into training & test set:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        print('Test set shape: {}'.format(X_test.shape))
        print()

        # Split train data into two parts
        train1, train2 = train_test_split(X_train.join(pd.DataFrame(y_train)),
                                          test_size=.5, random_state=42)

        # delete X_train and y_train variables as they will not be used
        del X_train, y_train
        print('Training set 1 shape: {}'.format(train1.shape))
        print()
        print('Training set 2 shape: {}'.format(train2.shape))
        print()

        # Fit the classifier
        text_model.fit(train1[text_feature], train1[label])

        # Predict test set labels
        train2['text_pred'] = text_model.predict(train2[text_feature])
        X_test['text_pred'] = text_model.predict(X_test[text_feature])

        # Fit Numeric Model Pipeline
        num_model.fit(train1[num_features], train1[label])

        # Predict test set labels
        train2['num_pred'] = num_model.predict(train2[num_features])
        X_test['num_pred'] = num_model.predict(X_test[num_features])

        # Stacked Model
        print('Stacked Model!')
        print(stacked_model)

        # Train 2nd level model on the Part 2 data
        stacked_model.fit(train2[['text_pred', 'num_pred']], train2[label])

        # Make stacking predictions on the test data
        X_test['stacking'] = stacked_model.predict(X_test[['text_pred', 'num_pred']])

        # Look at the model coefficients
        print('LogisticRegression Coefs: {}'.format(stacked_model.coef_))
        print()

        # see if model is overfitting
        print('Training Set Accuracy')
        print(stacked_model.score(train2[['text_pred', 'num_pred']], train2[label]))
        print()
        print('Test Set Accuracy')
        print(stacked_model.score(X_test[['text_pred', 'num_pred']], y_test))
        print()

        # append accuracy score to list: acc
        acc.append(stacked_model.score(X_test[['text_pred', 'num_pred']], y_test))

        # append accuracy score to list: f1
        f1.append(f1_score(y_test, X_test['stacking'], average='micro'))

        # Compute and print the confusion matrix and classification report
        print('Confusion matrix')
        print(confusion_matrix(y_test, X_test['stacking']))
        print()
        print('Classification report')
        print(classification_report(y_test, X_test['stacking']))
        print()
        print()

    # print average cross-validated metrics
    print('5-fold cross-validated Accuracy: {}'.format(np.mean(acc)))
    print()
    print('5-fold cross-validated F1 score: {}'.format(np.mean(f1)))
    print()

    # save stacked model for later
    with open("./models/lr_stack.pkl", 'wb') as model_file:
        pickle.dump(stacked_model, model_file)
