"""
*******************************************************************************************************************
model_utils.model_eval

This module contains customized utilities for training & evaluating Sentiment Analysis models:
    - split_df (splits DataFrame into KFold DataFrames)
    - model_training_metrics (iterate over a list of models, fitting them and getting evaluation metrics)
    - random_hyper_tune (performs hyper-parameter tuning)
    - text_feature_importance (Print a DataFrame with the Top N most important n_grams from the text model)
    - num_feature_importance (Print a DataFrame with most important features for tree-based models with numeric features)
    - stacked_model_metrics (fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics)

created: 12/31/19
last updated: 2/19/20
*******************************************************************************************************************
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score)
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


def model_training_metrics(models, df, features, label):
    """
    Splits DataFrame into 5-Fold partitions.
    Print model validation metrics for each partition and each model passed into the list models.
    Split each partition into training and test set,
    and return 5-Fold cross-validated average Accuracy and F1 metrics for multi-class classification

    @param models: list of models passed in for training and evaluation
    @param df: pandas DataFrame containing features and labels.
    @param features: pandas Series containing features, can be passed in as a list if specifying multiple features
    @param label: pandas Series containing labels
    """
    # split DataFrame into 5-Folds
    kfold_list = split_df(
        df=df,
        label_col=label
    )
    print('Split DataFrames into 5-Fold datasets!')
    print()

    # Start Text model training
    print('Starting Model Training!')
    print()

    # iterate over model list
    for model in models:
        print(model)
        print()

        # instantiate lists for loop
        acc = []
        f1 = []

        # iterate over list of KFold DataFrames
        for fold in kfold_list:
            # define feature set: X
            X = fold[features]
            # define label: y
            y = fold[label]

            # split each fold into training & test set:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
            print('Training set shape: {}'.format(X_train.shape))
            print()
            print('Test set shape: {}'.format(X_test.shape))
            print()

            # Fit the classifier
            model.fit(X_train, y_train)

            # Predict test set labels & probabilities
            y_pred = model.predict(X_test)

            # see if model is overfitting
            print('Training Set Accuracy')
            print(model.score(X_train, y_train))
            print()
            print('Test Set Accuracy')
            print(model.score(X_test, y_test))
            print()

            # append accuracy score to list: acc
            acc.append(model.score(X_test, y_test))

            # append accuracy score to list: f1
            f1.append(f1_score(y_test, y_pred, average=None))

            # Compute and print the confusion matrix and classification report
            print('Confusion matrix')
            print(confusion_matrix(y_test, y_pred))
            print()
            print('Classification report')
            print(classification_report(y_test, y_pred))
            print()

        # print 5-fold cross-validated Accuracy & F1 score
        print('5-fold cross-validated Accuracy: {}'.format(np.mean(acc)))
        print()
        print('5-fold cross-validated F1 score: {}'.format(np.mean(f1)))
        print()


def model_random_hyper_tune(df, features, label, model, param_grid, n_iters, n_folds, model_file_path):
    """
    Performs hyper-parameter tuning for a model using Randomized Search.
    In addition to specifying the pandas DataFrame, features, labels and model to cross-validate,
    specify the number of iterations and number of cross-validation folds.
    Finally specify the path to export the pickle file, which can be used for later.

    @param model: model to tune
    @param df: pandas DataFrame containing features and labels
    @param features: pandas Series containing features, can be passed in as a list if specifying multiple features
    @param label: pandas Series containing labels
    @param param_grid: dictionary of hyper-parameters to tune
    @param n_iters: nubmer of iterations to specify before training is ceased
    @param n_folds: number of cross-validation folds to partition pandas DataFrame
    @param model_file_path: path to export the model as a pickle file
    @return: return a tuple of the best performing model and a pandas DataFrame with all of the parameters and training/
        holdout results
    """

    # define feature set: X
    X = df[features]
    # define label: y
    y = df[label]

    # Create a random search object
    random_model = RandomizedSearchCV(
        estimator=model,
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
    print(dict(cv_results_df))
    print()

    # Extract and print the column with a dictionary of hyperparameters used
    column = cv_results_df.loc[:, ["params"]]
    print(dict(column))
    print()

    # Extract and print the row that had the best mean test score
    best_row = cv_results_df[cv_results_df["rank_test_score"] == 1]
    print(dict(best_row))
    print()

    # save model for later
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(random_model.best_estimator_, model_file)

    return random_model.best_estimator_, cv_results_df


def text_feature_importance(df, text_feature, vectorizer, top_n=20):
    """
    Print a pandas DataFrame with the Top N most important n_grams by the TfIdf Vectorizer weights,
    used as input features for the text model
    @param df: pandas DataFrame containing the text feature
    @param text_feature: pandas Series containing the text feature
    @param vectorizer: TfIdf Vectorizer that was used for feature engineering for text model
    @param top_n: Number of words to output
    """
    # fit vectorizer
    tfidf_matrix = vectorizer.fit_transform(df[text_feature])

    # get feature names
    features = vectorizer.get_feature_names()

    # Return the top n features that on average are most important amongst documents
    rows = np.squeeze(tfidf_matrix.toarray())

    # compute mean tfidf weights across documents
    tfidf_means = np.mean(rows, axis=0)

    # Get top n tfidf values in row and return them with their corresponding feature names
    topn_ids = np.argsort(tfidf_means)[::-1][:top_n]

    # return top features
    top_feats = [(features[i], tfidf_means[i]) for i in topn_ids]

    # create pandas DataFrame
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']

    # print pandas DataFrame
    print(df)


def num_feature_importance(df, model, features):
    """
    Return a pandas DataFrame with most important features for tree-based models with numeric features,
    sorted in descending order by feature importance.

    @param df: pandas DataFrame containing the features
    @param model: model that was used for training
    @param features: list of numeric features
    @return:
    """
    # Calculate feature importance: feature_importance
    feature_importance = model.estimators_[0].feature_importances_

    # Create a list of features: feature_list
    feature_list = list(df[features])

    # Save the results inside a DataFrame using feature_list as an index: relative_importance
    relative_importance = pd.DataFrame(index=feature_list, data=feature_importance, columns=["importance"])

    # print only features with relative importance higher than 1%
    print(dict(relative_importance[relative_importance['importance'] > 0.01].sort_values('importance', ascending=False)))


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
    Splits DataFrame into 5-Fold partitions.
    Fits both numeric, text and stacked model and prints model validation metrics for stacked models for each partition.
    Split each partition into training, test and holdout set,
    and return 5-Fold cross-validated average Accuracy and F1 metrics for multi-class classification

    @param df: pandas DataFrame containing features and labels
    @param label: pandas Series containing labels
    @param text_model: text model that will be used for training
    @param text_feature: pandas series that will be used as feature for text model
    @param num_model: model that will use numeric features for training
    @param num_features: list of pandas series that will be used as features for model with numeric features
    @param stacked_model: model that will be used in second layer of modeling pipeline
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
