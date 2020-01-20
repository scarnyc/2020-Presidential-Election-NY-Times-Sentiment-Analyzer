"""
********************************************************************************************************************
model_run.py

This script runs the model that was trained to analyze the sentiment of N.Y.T. articles via the following functions:
    - 

Please Note:
    The .csv files were created by calling the nyt_api.main module.

created: 12/31/19
last updated: 1/18/20
********************************************************************************************************************
"""
import pandas as pd
from core_utils.dataframe import union_csv
from custom_utils.clean_dataframe import preprocess_df
from model_utils.feature_eng import (date_feats, my_stopwords,
                                     tb_sentiment, nltk_sentiment,
                                     row_avg, sentiment_label, char_count,
                                     lemma_nopunc, split_x_y)
from graph_utils.graph import corr_heatmap
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from model_utils.model_eval import text_model_metrics, num_model_metrics, stacked_model_metrics
from sklearn.model_selection import train_test_split

# read trump .csv files & union them into a single DataFrame: trump_df
article_df = union_csv(
    csv_path=r'.\training_data',
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
print('Preprocessed DataFrame!')
print()
print('New DataFrame Shape: {}'.format(article_df.shape))
print()

# generate date_features for article_df
article_df = date_feats(article_df, 'pub_date')

# generate TextBlob sentiment and subjectivity scores for each row of 'text' in article_df: tb_sent_scores
# create a new DataFrame filled with scores: tb_sentiment_df
# join sentiment scores to original article_df DataFrame
tb_sent_scores = [tb_sentiment(text=row) for row in article_df['text']]
tb_sentiment_df = pd.DataFrame(tb_sent_scores)
article_df = article_df.join(tb_sentiment_df)
# delete unused variables
del tb_sent_scores, tb_sentiment_df
print('Generated TextBlob Sentiment Scores!')
print()
print(article_df[article_df['polarity'] == article_df['polarity'].max()]['text'].values)
print()

# generate Vader sentiment scores for each row of 'text' in article_df: nltk_sent_scores
# create a new DataFrame filled with scores: nltk_sentiment_df
# join sentiment scores to original article_df DataFrame
nltk_sent_scores = [nltk_sentiment(text=row) for row in article_df['text']]
nltk_sentiment_df = pd.DataFrame(nltk_sent_scores)
article_df = article_df.join(nltk_sentiment_df)
# delete unused variables
del nltk_sent_scores, nltk_sentiment_df
print('Generated Vader Sentiment Scores!')
print()
print(article_df[article_df['compound'] == article_df['compound'].max()]['text'].values)
print()

# compute the average of the Vader and TextBlob sentiment scores: article_df['sentiment']
article_df = row_avg(article_df, 'compound', 'polarity', 'sentiment')
print('Computed Mean Sentiment Scores!')
print()
print('Positive Sentiment')
print(article_df[article_df['sentiment'] == article_df['sentiment'].max()]['text'].values)
print()
print('Negative Sentiment')
print(article_df[article_df['sentiment'] == article_df['sentiment'].min()]['text'].values)
print()
print('Neutral Sentiment')
print(article_df[article_df['sentiment'] == 0]['text'].values)
print()

# compute the labels for modelling:
article_df = sentiment_label(
    df=article_df,
    col_for_label='sentiment',
    label_col='sentiment_label'
)
print('Computed labels for Modelling')
print()
print(article_df['sentiment_label'].unique())
print()

# filter out rows that have np.nan values
article_df = article_df[article_df['sentiment_label'] != 'nan']
print('Not Null DataFrame Metadata: {}'.format(article_df.info()))
print()
print(article_df['sentiment_label'].value_counts())
print()

# count number of characters:
article_df = char_count(
    df=article_df,
    text='text',
    new_pd_series='char_count'
)
print(article_df['char_count'].head())
print()

# lemmatize words & remove punctuation:
article_df['text_feat'] = article_df['text'].apply(lemma_nopunc)
print(article_df['text_feat'].head())
print()
print(article_df.columns)

# return column subsets of article_df
model_df = article_df[
    ['text_feat', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity', 'neg', 'neu', 'pos',
     'sentiment_label']
]

# plot heatmap of feature correlations
corr_heatmap(df=article_df,
             cols=['month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity', 'neg', 'neu', 'pos']
             )

# split DataFrame into training & testing sets: X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_x_y(
    df=article_df,
    features=['text_feat', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity', 'neg',
              'neu', 'pos'],
    label='sentiment_label',
    contains_col='text_feat',
    contains_term='trump',
    mapper={'positive': 1, 'neutral': 0, 'negative': -1}
)
print('Training features shape: {}'.format(X_train.shape))
print()
print('Test features shape: {}'.format(X_test.shape))
print()

# instantiate list of models: models
models = [
    MultinomialNB(),
    OneVsRestClassifier(LinearSVC(C=100, max_iter=1000000, random_state=42, class_weight='balanced', n_jobs=4)),
    OneVsRestClassifier(RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42, n_jobs=4,
                                               class_weight='balanced')),
    OneVsRestClassifier(xgb.XGBClassifier(n_jobs=4, random_state=42)),
    OneVsRestClassifier(
        LogisticRegression(C=100, max_iter=5000, solver='liblinear', n_jobs=4,
                           random_state=42, class_weight='balanced'))
]
print('Instantiated models!')
print()

# print out text model metrics
text_model_metrics(
    models=models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    text_feature='text_feat',
    n_gram_range=(2, 3),
    k=1200,
    stopwords=my_stopwords
)

# print out numeric model metrics
num_model_metrics(
    models=models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    num_features=['month', 'day', 'dayofweek', 'hour', 'char_count', 'word_count', 'subjectivity',
                  'neg', 'neu', 'pos']
)

# Split train data into two parts
train1, train2 = train_test_split(X_train.join(pd.DataFrame(y_train)), test_size=.5, random_state=42)

# print out stacked model metrics
stacked_model_metrics(
    train1_df=train1,
    train2_df=train2,
    test_df=X_test,
    y_test=y_test,
    label_col='sentiment_label',
    text_model=OneVsRestClassifier(LinearSVC(C=100, max_iter=1000000, random_state=42,
                                             class_weight='balanced', n_jobs=4)),
    text_feature='text_feat',
    text_prediction_col='text_pred',
    n_gram_range=(2, 3),
    k=1200,
    stopwords=my_stopwords,
    text_model_pkl="./models/text_pipe_svm.pkl",
    num_model=OneVsRestClassifier(xgb.XGBClassifier(n_jobs=4, random_state=42)),
    num_train1_features=train1[['month', 'day', 'dayofweek', 'hour', 'char_count', 'word_count', 'subjectivity',
                                'neg', 'neu', 'pos']],
    num_features=['month', 'day', 'dayofweek', 'hour', 'char_count', 'word_count', 'subjectivity',
                  'neg', 'neu', 'pos'],
    num_prediction_col='num_pred',
    num_model_pkl="./models/num_pipe_xgb.pkl",
    stacked_model=OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear', n_jobs=4,
                                                         random_state=42, class_weight='balanced')),
    stacked_features=['text_pred', 'num_pred'],
    stacked_model_pkl="./models/lr_stack.pkl"
)
