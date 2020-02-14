# nyt_sentiment_analyzer
Sentiment Analysis of N.Y. Times Articles about 2020 U.S. Democratic Presidential Candidates

This script performs EDA on data & trains a stacked machine learning model to perform sentiment analysis of U.S. Presidential Democratic Candidates,
    using N.Y. Times article data via the following functions:

    - union_csv (reads .csv files containing Article abstract descriptions of N.Y. Times articles into a single pandas DataFrame.
        Please note that the .csv files were created by calling the nyt_api.main module)
    - preprocess_df (pre-processes aforementioned DataFrame by dropping null values and duplicates,
        and also concatenating columns for text feature engineering)
    - date_feats (generates year, month, day and hour date features from existing date feature)
    - filter_dataframe (filters out articles prior to a user-defined year)
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
    - num_feature_importance (print the most important feautres for tree-based models with numeric features)
    - stacked_model_metrics (finally, fit both the numeric & text models that were previously tuned performing cross-validation,
        and adding a second-layer LogisticRegression stacked model that will use the predictions from the other two models as features,
        for the final predictions)
        
 The stacked model achieves about 58% accuracy and F1, which is better than a random guess. 
 
 Next Step - 
    RNN & LSTM 
    
 More to come...
