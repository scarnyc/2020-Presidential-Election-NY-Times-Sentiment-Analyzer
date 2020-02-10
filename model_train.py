"""
********************************************************************************************************************
model_train.py

This script performs EDA on data & trains a model to analyze the sentiment of N.Y. Times articles via the following functions:
    -

Please Note:
    The .csv files were created by calling the nyt_api.main module.

created: 12/31/19
last updated: 1/28/20
********************************************************************************************************************
"""
import pandas as pd
from core_utils.dataframe import union_csv
from custom_utils.clean_dataframe import preprocess_df, filter_dataframe
from model_utils.feature_eng import (date_feats, my_stopwords, tb_sentiment,
                                     sentiment_label, lemma_nopunc)
from graph_utils.graph import corr_heatmap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from model_utils.model_eval import (text_model_metrics, num_model_metrics, num_feature_importance,
                                    text_random_hyper, num_random_hyper, stacked_model_metrics,
                                    text_feature_importance)
from graph_utils.graph import (plot_word_freq, two_dim_tf_viz,
                               time_series_line_viz)


# read .csv files & union them into a single DataFrame: trump_df
article_df = union_csv(
    csv_path=r'.\data',
    glob_pattern='*.csv'
)

# create 'text' feature
# drop null rows from the 'abstract' column
# return preprocessed DataFrame with 'headline_main','web_url','text','word_count','pub_date' columns
article_df = preprocess_df(
    df=article_df,
    new_col='text',
    con_col1='headline_main',
    con_col2='abstract',
    subset_list=['abstract'],
    filter_col_list=['headline_main', 'web_url', 'text', 'word_count', 'pub_date']
)

# generate date_features for article_df
article_df = date_feats(article_df, 'pub_date')

# generate TextBlob sentiment and subjectivity scores for each row of 'text' in article_df: tb_sent_scores
# create a new DataFrame filled with scores: tb_sentiment_df
# join sentiment scores to original article_df DataFrame
tb_sent_scores = [tb_sentiment(text=row) for row in article_df['text']]
tb_sentiment_df = pd.DataFrame(tb_sent_scores)
article_df = article_df.merge(tb_sentiment_df, left_index=True, right_index=True)
# delete unused variables
# del tb_sent_scores, tb_sentiment_df
print('Generated TextBlob Sentiment Scores!')
print()
print(article_df[article_df['polarity'] == article_df['polarity'].max()]['text'].values)
print()

# compute the labels for modelling: article_df['sentiment_label']
article_df = sentiment_label(
    df=article_df,
    col_for_label='polarity',
    label_col='sentiment_label'
)

# count number of characters: article_df['char_count']
article_df = char_count(
    df=article_df,
    text='text',
    new_pd_series='char_count'
)

# lemmatize words & remove punctuation: article_df['text_feat']
article_df['text_feat'] = article_df['text'].apply(lemma_nopunc)
print(article_df['text_feat'].head())
print()
print(article_df.columns)
print()

# return column subsets of article_df for modeling: model_df
model_df = article_df[
    ['text_feat', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity', 'char_count', 'sentiment_label']
]

# drop char_count variable function
# plot heatmap of feature correlations
corr_heatmap(df=article_df,
             cols=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity']
             )

# # plot word frequencies
# plot_word_freq(
#     pd_series=article_df[article_df['sentiment_label'] == 'positive']['text'],
#     plot_title='Trump BOW FREQUENCY',
#     n=30
# )

# # plot tfidf Scatter plot
# two_dim_tf_viz(
#     df=article_df,
#     pd_series='text',
#     pd_color_series='sentiment_label',
#     pd_hover_series='headline_main',
#     max_features=800,
#     plot_title='N.Y. Times Trump Articles Sentiment Clusters'
# )

# # plot Time Series Line plot
# time_series_line_viz(
#     df=article_df,
#     date_index='pub_date',
#     pd_series='sentiment',
#     plot_title='N.Y. Times Trump Articles Avg. Daily Sentiment'
# )

# # instantiate list of models: models
# models = [
#     OneVsRestClassifier(MultinomialNB()),
#     OneVsRestClassifier(LinearSVC(C=1000, max_iter=5000, random_state=42, class_weight='balanced')),
#     OneVsRestClassifier(RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42, n_jobs=4,
#                                                class_weight='balanced')),
#     OneVsRestClassifier(XGBClassifier(n_jobs=4, random_state=42)),
#     OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
#                                            random_state=42, class_weight='balanced'))
# ]
# print('Instantiated models!')
# print()

# # instantiate list of vectorizers: vectorizers
# vectorizers = [
#     CountVectorizer(max_features=200, ngram_range=(2, 3), stop_words=my_stopwords),
#     TfidfVectorizer(max_features=200, ngram_range=(2, 3), stop_words=my_stopwords)
# ]
# print('Instantiated vectorizers!')
# print()

# # print out text model metrics
# text_model_metrics(
#     models=models,
#     vectorizers=vectorizers,
#     df=model_df,
#     label='sentiment_label',
#     text_feature='text_feat'
# )

# # print out numeric model metrics
# num_model_metrics(
#     models=models,
#     df=model_df,
#     label='sentiment_label',
#     num_features=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity']
# )

# # tune hyper parameters for text model: text_pipe
# text_pipe = text_random_hyper(
#     df=model_df,
#     text_feature='text_feat',
#     label='sentiment_label',
#     model=OneVsRestClassifier(XGBClassifier(random_state=42)),
#     vectorizer=TfidfVectorizer(stop_words=my_stopwords),
#     n_iters=15,
#     n_folds=5
# )

# get feature importances from TFIDF scores: tfidf_df
tfidf_df = text_feature_importance(df=model_df, text_feature='text_feat', vectorizer=text_pipe[0])

# tune hyper parameters for model with numeric feature inputs: num_pipe
num_pipe = num_random_hyper(
    df=model_df,
    num_features=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity'],
    label='sentiment_label',
    model=OneVsRestClassifier(XGBClassifier(random_state=42)),
    n_iters=15,
    n_folds=5
)

# look at most important features for text model
num_feat_df = num_feature_importance(df=model_df,
                                     model=num_pipe[1],
                                     features=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity'])

# print out stacked model metrics
stacked_model_metrics(
    df=model_df,
    label='sentiment_label',
    text_model=Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 3), stop_words=my_stopwords)),
                     ('scaler', StandardScaler(with_mean=False)),
                     ('dim_red', SelectKBest(chi2, k=200)),
                     ('clf', OneVsRestClassifier(XGBClassifier(n_estimators=1000, min_samples_leaf=48, max_depth=3,
                                                               learning_rate=1.2476510067114095, colsample_bytree=0.3,
                                                               booster='gbtree', random_state=42)))
                     ]),
    text_feature='text_feat',
    text_prediction='text_pred',
    text_model_pkl="./models/text_pipe_xgb.pkl",
    num_model=num_pipe,
    num_features=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity'],
    num_prediction='num_pred',
    num_model_pkl="./models/num_pipe_xgb.pkl",
    stacked_model=OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
                                                         random_state=42, class_weight='balanced')),
    stacked_model_pkl="./models/lr_stack.pkl"
)
