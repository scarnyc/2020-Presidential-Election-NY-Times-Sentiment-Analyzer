"""
*******************************************************************************************************************
model_utils.model_eval

This package contains customized utilities for training Sentiment Analysis models:
    - split_df (splits DataFrame into KFold DataFrames)
    - text_model_metrics (iterate over a list of models, fitting them and getting evaluation metrics on text data)
    - num_model_metrics(iterate over a list of models, fitting them and getting evaluation metrics on numeric data)
    - text_grid_hyper(performs hyper-parameter tuning for a text model)
    - stacked_model_metrics (fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics)

created: 1/5/20
last updated: 1/28/20
*******************************************************************************************************************
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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


def text_random_hyper(df, text_feature, label, model, vectorizer, n_iters, n_folds):
    """
    Performs hyper-parameter tuning for a text model.

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
                     ('feature_select', SelectKBest(chi2)),
                     ('clf', model)
                     ])
    print(pipe)
    print()

    # Create the parameter grid
    param_grid = {
        'vectorizer__ngram_range': [(1, 3), (2, 3)],
        'feature_select__k': [50, 100, 200],
        'clf__estimator__booster': ['gbtree', 'gblinear', 'dart'],
        'clf__estimator__colsample_bytree': [0.3, 0.7],
        'clf__estimator__n_estimators': [100, 500, 1000],
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

    return random_model.best_estimator_


def num_random_hyper(df, num_features, label, model, n_iters, n_folds):
    """
    Performs hyper-parameter tuning for a model with numeric feature-inputs.
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

    # instantiate model training pipeline
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
        'clf__estimator__n_estimators': [100, 500, 1000],
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

    return random_model.best_estimator_


def stacked_model_metrics(
        df,
        label,
        text_model,
        text_feature,
        text_prediction,
        text_model_pkl,
        num_model,
        num_features,
        num_prediction,
        num_model_pkl,
        stacked_model,
        stacked_model_pkl
):
    """
    Fits models to text & num data, plus adds stacked model ensemble, and gets evaluation metrics.

    """
    # split DataFrame into 5-Folds
    kfold_list = split_df(
        df=df,
        label_col=label
    )
    print('Split DataFrames into 5-Fold datasets!')
    print()

    # Text Model Pipeline
    print('Stacked Model Training!')
    print('Text Model!')
    print(text_model)
    print()

    # instantiate lists for loop
    acc = []
    f1 = []

    # iterate over list of KFold DataFrames
    for fold in kfold_list:
        # define feature set: X
        X = fold.drop(label, axis=1)
        # define label: y
        y = fold[label]

        # split each fold into training & test set:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
        print('Test set shape: {}'.format(X_test.shape))
        print()

        # Split train data into two parts
        X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, stratify=y,
                                                                  test_size=.5, random_state=42)
        del X_train, y_train
        print('Training set 1 shape: {}'.format(X_train1.shape))
        print()
        print('Training set 2 shape: {}'.format(X_train2.shape))
        print()

        # Fit the classifier
        text_model.fit(X_train1[text_feature], y_train1)

        # Predict test set labels
        X_train2[text_prediction] = text_model.predict(X_train2[text_feature])
        X_test[text_prediction] = text_model.predict(X_test[text_feature])


        # Numeric Model Pipeline
        print('Numeric Model!')
        print(num_model)
        print()

        # Fit the classifier
        num_model.fit(X_train1[num_features], y_train1)

        # Predict test set labels
        X_train2[num_prediction] = num_model.predict(X_train2[num_features])
        X_test[num_prediction] = num_model.predict(X_test[num_features])


        # Stacked Model
        print('Stacked Model!')
        print(stacked_model)

        # Train 2nd level model on the Part 2 data
        stacked_model.fit(X_train2[[text_prediction, num_prediction]], y_train2)

        # Make stacking predictions on the test data
        X_test['stacking'] = stacked_model.predict(X_test[[text_prediction, num_prediction]])

        # Look at the model coefficients
        print('LogisticRegression Coefs: {}'.format(stacked_model.coef_))
        print()
        # see if model is overfitting
        print('Training Set Accuracy')
        print(stacked_model.score(X_train2[[text_prediction, num_prediction]], y_train2))
        print()
        print('Test Set Accuracy')
        print(stacked_model.score(X_test[[text_prediction, num_prediction]], y_test))
        print()

        # append accuracy score to list: acc
        acc.append(stacked_model.score(X_test[[text_prediction, num_prediction]], y_test))

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

    print('5-fold cross-validated Accuracy: {}'.format(np.mean(acc)))
    print()
    print('5-fold cross-validated F1 score: {}'.format(np.mean(f1)))
    print()

    # save text model for later
    with open(text_model_pkl, 'wb') as model_file:
        pickle.dump(text_model, model_file)

    # save model for later
    with open(num_model_pkl, 'wb') as model_file:
        pickle.dump(num_model, model_file)

    # save model for later
    with open(stacked_model_pkl, 'wb') as model_file:
        pickle.dump(stacked_model, model_file)
