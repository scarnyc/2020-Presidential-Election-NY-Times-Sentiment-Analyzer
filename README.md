# Sentiment Analysis of N.Y. Times Articles about 2020 U.S. Presidential Candidates

## Experiment Abstract: Application of Sentiment Analysis on Presidential Candidates using the N.Y. Times

This python-based Sentiment Analysis project helps perform sentiment analysis of U.S. Presidential Candidates, based on content from N.Y. Times articles.  I'm an avid reader of the N.Y.Times and after reading numerous articles about presidential polling results, I wondered what is the general sentiment of these candidates based on the content in the articles that iswritten about them?

Now we know the N.Y. Times is politically left-leaning, so there is inherent bias going into this exercise.  However,  the availability of data via the N.Y.Times Article Search API, the journalistic quality of their writing as well as the the analysis of political elections still make the N.Y. Times a viable option as a data source for this experiment.

The package performs EDA on data & trains a stacked machine learning model to perform sentiment analysis of U.S. Presidential Candidates, 
using the N.Y. Times Article Search API, so it uses 11,000 N.Y. Times article abstracts as the text to predict sentiment.  This is a multi-class classification problem, predicting positive, neutral and negative sentiment using Natural Language Processing techniques to pre-process the text and engineer features.

## So What?

The sentiment analysis model that I created for this project achieves a 60% harmonic mean of precision & recall (F1 score) and it is predicting that Bernie Sanders has the highest average sentiment prediction, while Donald Trump has the lowest. Now the data from the N.Y. Times was run through February 2020 and the results displayed here are in line with the results from the Iowa and New Hampshire primaries.  Please refer to the "Findings & Results" section below for more information on the relevance of the findings.

## What does the Sentiment Analysis Pipeline technically do?

At a high level, the sentiment_analysis_pipe() function in the nyt_sentiment_analyzer.py script will perform the following functions:
    1) Read in data from multiple N.Y.Times .csv files from a specified directory into a single DataFrame. Files contain data about N.Y.        Times Articles about U.S. Presidential Candidates scraped from the Article Search API. 
    2) Then it will pre-process the data, engineer additional features, as well as label the data using TextBlob sentiment analysis     scores.
    3) Next, the function will generate a series of graphs to assist with EDA in the Data Science workflow.
    4) After that, it will train user-selected Sci-kit Learn Models and print out F1 and Accuracy score metrics, performing 5-Fold cross-        validation. This allows the user to compare models with numeric & text-based features and see how they perform, and see if the model is overfitting to the training data.
    5) Then the function will both tune the hyper-parameters of both models (numeric & text-based features) and print the most important        features for each model type, respectively.
    6) Next, the function will train and evaluate a stacked model pipeline: 
        Using the aforementioned predictions from the text-based model and the numeric model as features, it will train a second-layered         Logistic Regression model.
    7) Finally, the pipeline will make predictions on the data and produces a final graph showing the average sentiment of the N.Y. Times        Articles about a particular candidate over time.

## Model Training Methodology:
The model training methodology for this project takes a two-step approach:
1) Train text models on BOW and TfIdf scores using content from the articles via the N.Y. Times Article Search API.  All of the text models that were trained were evaluated using Bi-Gram Bag of Words (BOW) Frequencies and TfIdf Scores as features for text-based models. 
    Standardization was only performed on the TfIdf scores and dimensionality reduction using Chi-Square test was performed on both text-based feature types.
2) Train numeric models using date features, character counts, article total word counts, and TextBlob subjectivity.  All numeric models that were trained were evaluated using Min-Max Scaling as features prior to model fitting. 

## Models Trained & Evaluated (for both Text & Numeric Models):
- Logistic Regression
- XGBoost Classifier
- Random Forest Classifier
- Linear SVM Classifier
- Multinomial Naiive Bayes Classifier
- Recurrent Neural Network (LSTM)

## Findings & Results:
### Model Results
- For multi-class classification F1 Score was used to evaluate data with imbalanced classes   
- XGBoost Classifier outperformed the other models for the text-based model with 45.7% F1 score using Bi-gram TfIdf weights as features.   
- XGBoost Classifier also outperformed the other models using numeric features, touting 66% F1 Score.       
- The stacked Logistic Regression model achieves 60% F1 Score, which is 10 percentage points better than a random guess. 
    However, while the numeric features involved in the model stacking significantly improve F1 Score (by approximately 15 percentage points), 
    F1 Score is ill-defined for negative predictions, with no prediction samples. 
    This finding is conclusive with the results for the XGBoost Classifier text model.
- The LSTM had 68% F1 Score, however it only predicted 2/3 classes.  The LSTM Classifier needed more training data.
     
### Sentiment Score Prediction Interpretation
While the model results suggest that the model has a 60% harmonic mean of precision & recall, the interpretation of the sentiment score predictions are quite interesting.

Consider the table below that shows the results for candidates and their average sentiment predictions from the model:  

|candidate         | avg. predicted sentiment |
|------------------|:------------------------:|
|Bernie Sanders    |   0.56                   |
|Amy Klobuchar     |   0.50                   |
|Elizabeth Warren  |   0.43                   |
|Joe Biden         |   0.18                   |
|Donald Trump      | -0.02                    |
 
The model is predicting that Bernie Sanders has the highest average sentiment prediction while Donald Trump has the lowest.  This prediction of Senator Sanders is in line with current polls and Caucus results from Iowa & New Hampshire (which was the around the time the data from the N.Y. Times Article Search API data was last run).  It would be interesting to run the data throughout the election to current date, and see if the results wind up with Joe Biden having the highest average sentiment score.  On the other hand, it's interesting that President Trump has the lowest sentiment, out of any Presidential Candidate.  The N.Y. Times has a reputation for being a progressive news organization. It's still interesting that compared to the rest of the candidates Trump's average sentiment is the lowest, considering that the President's impeachment trial was happening during the same time these articles were collected.

## Next Steps
- Train the model on a wider universe of N.Y. Times articles (through current) to accurately predict the election outcome
- Use negation rules to generate intelligent labels
- Train the LSTM on a larger set of data
