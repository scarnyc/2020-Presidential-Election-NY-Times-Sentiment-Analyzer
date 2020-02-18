# nyt_sentiment_analyzer .py package
Sentiment Analysis of N.Y. Times Articles about 2020 U.S. Democratic Presidential Candidates

This script performs EDA on data & trains a stacked machine learning model to perform sentiment analysis of U.S. Presidential Democratic Candidates, 
using the N.Y. Times Article Search API. Ths is a multi-class classification problem, predicting positive, neutral and negative sentiment.
At a high level, the sentiment_analysis_pipe() function in the nyt_sentiment_analyzer.py script will perform the following functions:
    1) Read in data from multiple N.Y.Times .csv files from a specified directory into a single DataFrame.
        Files contain data about N.Y. Times Articles about U.S. Presidential Democratic Candidates scraped from the Article Search API 
    2) Then it will pre-process the data, engineer additional features, as well as label the data using TextBlob sentiment analysis scores.
    3) Next, the function will generate a series of graphs to assist with EDA in the Data Science workflow.
    4) After that, it will train user-selected Sci-kit Learn Models and print out F1 and Accuracy score metrics, performing 5-Fold cross-validation.
        This allows the user to compare models with numeric & text-based features and see how they perform, and see if the model is overfitting to the training data.
    5) Then the function will both tune the hyper-parameters of both models (numeric & text-based features) and print the most important features for each model type, respectively.
    6) Next, the function will train and evaluate a stacked model pipeline: 
        Using the aforementioned predictions from the text-based model and the numeric model as features, it will train a second-layered Logistic Regression model.
    7) Finally, the pipeline will make predictions on the data and produce one last graph showing the average sentiment of the N.Y. Times Articles 
        about a particular candidate over time.

Models Trained & Evaluated for both Text & Numeric Models:
- Logistic Regression
- XGBoost Classifier
- Random Forest Classifier
- Linear SVM Classifier
- Multi-nomial Naiive Bayes Classifier

Methods for training models:
- All text models that were trained were evaluated using Bi-Gram Bag of Words (BOW) Frequencies and TfIdf Scores as features for text-based models. 
    Standardization was only performed on the TfIdf scores and dimensionality reduction using Chi-Square test was performed on both text-based feature types.
- All numeric models that were trained were evaluated using Min-Max Scaling as features prior to model fitting. 
    The numeric models used date features, article abstract character counts, article word counts, and TextBlob subjectivity. 

Initial Findings:
- For multi-class classification Micro F1 Score was used to evaluate data with imbalanced classes   
- XGBoost Classifier outperformed the other models for the text-based model with 45.7% F1 score using Bi-gram TfIdf weights as features.   
- XGBoost Classifier also outperformed the other models using numeric features, touting 66% F1 Score.       
- The stacked Logistic Regression model achieves 60% F1 Score, which is 10 percentage points better than a random guess. 
    However, while the numeric features involved in the model stacking significantly improve F1 Score (by approximately 15 percentage points), 
    F1 Score is ill-defined for negative predictions, with no prediction samples. 
    This finding is conclusive with the results for the XGBoost Classifier text model. 
 
Next Steps:
 - Add rolling window averages for Time Series Plot
 - Add RNN & LSTM to model training pipeline
 - Add logging functionality
 - Try using LabelBinarizer() for one-hot encoding of Labels: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
    
 More to come...