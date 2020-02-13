"""
********************************************************************************************************************
model_train.py

This script performs EDA on data & trains a model to analyze the sentiment of N.Y. Times articles via the following functions:
    -

Please Note:
    The .csv files were created by calling the nyt_api.main module.

created: 12/31/19
last updated: 2/13/20
********************************************************************************************************************
"""
from core_utils.dataframe import union_csv
from custom_utils.clean_dataframe import preprocess_df, filter_dataframe
from model_utils.feature_eng import (date_feats, my_stopwords,
                                     char_count, sentiment_label, lemma_nopunc,
                                     apply_func, drop_high_corr, sentiment_analyzer)
from graph_utils.graph import (corr_heatmap, plot_word_freq, two_dim_tf_viz,
                               time_series_line_viz)
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
from model_utils.model_eval import (text_model_metrics, num_model_metrics,
                                    num_feature_importance, text_random_hyper_tune,
                                    num_random_hyper_tune,
                                    stacked_model_metrics, text_feature_importance)


# read .csv files & union them into a single DataFrame: trump_df
article_df = union_csv(
    csv_path=r'.\data',
    glob_pattern='*.csv'
)

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
article_df = date_feats(df=article_df, date_col='pub_date')

# filter out articles written prior to Democratic election campaigns (ie. 2019)
article_df = filter_dataframe(df=article_df, col_to_filter='year', value_to_filter=2016)

# generate TextBlob sentiment and subjectivity scores for each row of 'text' in article_df: tb_sent_scores
article_df = sentiment_analyzer(df=article_df, text_feature='text')

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

# add list comprehension to remove digits in lemma_nopunc function
# lemmatize words & remove punctuation: article_df['text_feat']
article_df = apply_func(
    df=article_df,
    pd_series='text',
    new_pd_series='text_feat',
    func=lemma_nopunc
)

# return column subsets of article_df for modeling: model_df
model_df = article_df[
    ['text_feat', 'year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'subjectivity', 'char_count',
     'sentiment_label']
]

# # plot heatmap of feature correlations
# corr_heatmap(df=article_df,
#              features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity']
#              )
#
# # automatically remove highly correlated features
# model_df = drop_high_corr(df=model_df)

# # plot word frequencies
# plot_word_freq(
#     pd_series=article_df[article_df['sentiment_label'] == 'positive']['text'],
#     plot_title='BOW FREQUENCY',
#     stopwords=my_stopwords,
#     n=30
# )

# # plot tfidf Scatter plot
# two_dim_tf_viz(
#     df=article_df,
#     pd_series='text',
#     stopwords=my_stopwords,
#     pd_color_series='sentiment_label',
#     max_features=1000,
#     plot_title='N.Y. Times Article Sentiment Clusters'
# )

# # plot Time Series Line plot
# time_series_line_viz(
#     df=article_df,
#     date_index='pub_date',
#     pd_series='polarity',
#     plot_title='N.Y. Times Articles Avg. Daily Sentiment'
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
#     CountVectorizer(ngram_range=(1, 3), stop_words=my_stopwords),
#     TfidfVectorizer(ngram_range=(1, 3), stop_words=my_stopwords)
# ]
# print('Instantiated vectorizers!')
# print()

# # print out text model metrics
# text_model_metrics(
#     df=model_df,
#     models=models,
#     vectorizers=vectorizers,
#     text_feature='text_feat',
#     label='sentiment_label')

# # print out numeric model metrics
# num_model_metrics(
#     models=models,
#     df=model_df,
#     num_features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
#     label='sentiment_label'
# )

# # tune hyper parameters for text model: text_pipe, text_cv_results_df, text_model_params
# text_pipe, text_cv_results_df, text_model_params = text_random_hyper_tune(
#     df=model_df,
#     text_feature='text_feat',
#     label='sentiment_label',
#     model=OneVsRestClassifier(XGBClassifier(random_state=42)),
#     vectorizer=TfidfVectorizer(stop_words=my_stopwords),
#     n_iters=15,
#     n_folds=5,
#     model_type='text'
# )

# # comment: correct problem with text_feature_importance func
# # get feature importances from TFIDF scores: tfidf_df
# tfidf_df = text_feature_importance(df=model_df, text_feature='text_feat', vectorizer=text_pipe[0])

# # tune hyper parameters for model with numeric feature inputs: num_pipe, cv_results_df, model_params
# num_pipe, num_cv_results_df, num_model_params = num_random_hyper_tune(
#     df=model_df,
#     num_features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
#     label='sentiment_label',
#     model=OneVsRestClassifier(XGBClassifier(random_state=42)),
#     n_iters=15,
#     n_folds=5
# )

# # look at most important features for text model
# num_feat_df = num_feature_importance(df=model_df,
#                                      model=num_pipe[1],
#                                      features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count',
#                                                'subjectivity'])

# print out stacked model metrics
stacked_model_metrics(
    df=model_df,
    label='sentiment_label',
    text_model=Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 3), stop_words=my_stopwords)),
                     ('scaler', StandardScaler(with_mean=False)),
                     ('dim_red', SelectKBest(chi2, k=300)),
                     ('clf', OneVsRestClassifier(XGBClassifier(n_estimators=1000, min_samples_leaf=52, max_depth=20,
                                                               learning_rate=1.4434343434343435, colsample_bytree=0.7,
                                                               booster='gbtree', random_state=42)))
                     ]),
    text_feature='text_feat',
    # text_model_pkl="./models/text_pipe_xgb.pkl",
    num_model=Pipeline([('scaler', MinMaxScaler()),
                        ('clf', OneVsRestClassifier(XGBClassifier(n_estimators=1000, min_samples_leaf=44, max_depth=10,
                                                                  learning_rate=0.8395973154362416,
                                                                  colsample_bytree=0.7, booster='gbtree',
                                                                  random_state=42)))
                        ]),
    num_features=['year', 'month', 'day', 'dayofweek', 'hour', 'word_count', 'char_count', 'subjectivity'],
    # num_model_pkl="./models/num_pipe_xgb.pkl",
    stacked_model=OneVsRestClassifier(LogisticRegression(C=100, max_iter=5000, solver='liblinear',
                                           random_state=42, class_weight='balanced'))
    # stacked_model_pkl="./models/lr_stack.pkl"
)


