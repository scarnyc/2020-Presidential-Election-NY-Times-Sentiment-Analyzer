"""
********************************************************************************************************************
model_train.py

This script trains a model to analyze the sentiment of N.Y.T. articles via the following functions:
    -

Please Note:
    The .csv files were created by calling the nyt_api.main module.

created: 12/31/19
last updated: 1/28/20
********************************************************************************************************************
"""
import pandas as pd
from core_utils.dataframe import union_csv, filter_dataframe
from custom_utils.clean_dataframe import preprocess_df
from model_utils.feature_eng import (date_feats, my_stopwords, tb_sentiment,
                                     sentiment_label, lemma_nopunc)
from graph_utils.graph import corr_heatmap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from model_utils.model_eval import (text_model_metrics, num_model_metrics, text_random_hyper)
# , stacked_model_metrics
from sklearn.model_selection import train_test_split

# read .csv files & union them into a single DataFrame: trump_df
article_df = union_csv(
    csv_path=r'.\data',
    glob_pattern='*.csv'
)
print('DataFrame Sample')
print(article_df.head())
print()
print('DataFrame Shape: {}'.format(article_df.shape))
print()

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

# filter out articles that don't contain last name of candidates
article_df = filter_dataframe(df=article_df,
                              col='text',
                              contains_list=['Klobuchar', 'Sanders', 'Booker', 'Trump', 'Warren', 'Biden', 'Delaney',
                                             'Harris', 'Bennet', 'Bloomberg', 'Gabbard']
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

# compute the labels for modelling:
article_df = sentiment_label(
    df=article_df,
    col_for_label='polarity',
    label_col='sentiment_label'
)

# lemmatize words & remove punctuation:
article_df['text_feat'] = article_df['text'].apply(lemma_nopunc)
print(article_df['text_feat'].head())
print()
print(article_df.columns)

# return column subsets of article_df
model_df = article_df[
    ['text_feat', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity', 'sentiment_label']
]

# plot heatmap of feature correlations
corr_heatmap(df=article_df,
             cols=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity']
             )

# instantiate list of models: models
# models = [
#     OneVsRestClassifier(MultinomialNB()),
#     OneVsRestClassifier(LinearSVC(C=1000, max_iter=1000000, random_state=42, class_weight='balanced')),
#     OneVsRestClassifier(RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42, n_jobs=4,
#                                                class_weight='balanced')),
#     OneVsRestClassifier(xgb.XGBClassifier(n_jobs=4, random_state=42)),
#     OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
#                                            random_state=42, class_weight='balanced'))
# ]
# print('Instantiated models!')
# print()

# instantiate list of vectorizers: vectorizers
# vectorizers = [
#     CountVectorizer(max_features=200, ngram_range=(2, 3), stop_words=my_stopwords),
#     TfidfVectorizer(max_features=200, ngram_range=(2, 3), stop_words=my_stopwords)
# ]
# print('Instantiated vectorizers!')
# print()

# print out text model metrics
# text_model_metrics(
#     models=models,
#     vectorizers=vectorizers,
#     df=model_df,
#     label='sentiment_label',
#     text_feature='text_feat'
# )

# print out numeric model metrics
# num_model_metrics(
#     models=models,
#     df=model_df,
#     label='sentiment_label',
#     num_features=['month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity']
# )

# tune hyper parameters for text model
text_pipe = text_random_hyper(
    df=model_df,
    text_feature='text_feat',
    label='sentiment_label',
    model=OneVsRestClassifier(xgb.XGBClassifier(random_state=42)),
    vectorizer=TfidfVectorizer(ngram_range=(2, 3), stop_words=my_stopwords),
    n_iters=2,
    cfv=2
)

# Split train data into two parts
# train1, train2 = train_test_split(X_train.join(pd.DataFrame(y_train)), test_size=.5, random_state=42)

# print out stacked model metrics
# stacked_model_metrics(
#     train1_df=train1,
#     train2_df=train2,
#     test_df=X_test,
#     y_test=y_test,
#     label_col='sentiment_label',
#     text_model=OneVsRestClassifier(LinearSVC(C=100, max_iter=1000000, random_state=42, class_weight='balanced')),
#     text_feature='text_feat',
#     text_prediction_col='text_pred',
#     n_gram_range=(2, 3),
#     k=1200,
#     stopwords=my_stopwords,
#     text_model_pkl="./models/text_pipe_svm.pkl",
#     num_model=OneVsRestClassifier(xgb.XGBClassifier(n_jobs=4, random_state=42)),
#     num_train1_features=train1[['month', 'day', 'dayofweek', 'hour', 'char_count', 'word_count', 'subjectivity',
#                                 'neg', 'neu', 'pos']],
#     num_features=['month', 'day', 'dayofweek', 'hour', 'char_count', 'word_count', 'subjectivity',
#                   'neg', 'neu', 'pos'],
#     num_prediction_col='num_pred',
#     num_model_pkl="./models/num_pipe_xgb.pkl",
#     stacked_model=OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
#                                                          random_state=42, class_weight='balanced')),
#     stacked_features=['text_pred', 'num_pred'],
#     stacked_model_pkl="./models/lr_stack.pkl"
# )
