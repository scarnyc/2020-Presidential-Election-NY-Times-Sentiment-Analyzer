'''
********************************************************************************************************************
graph_eda.py

This script generates graphs for exploratory data analysis of N.Y. Times articles for sentiment analysis:
- Article Cluster Scatter Plot
- Avg. Daily Sentiment Line Plot
- Bar Graph of Word Frequencies

Please Note:
    The .csv files were created by calling the nyt_api.main module.

created: 12/31/19
last updated: 1/4/20
********************************************************************************************************************
'''
import pandas as pd
from core_utils.dataframe import union_csv
from custom_utils.clean_dataframe import preprocess_df
from model_utils.feature_eng import (date_feats, sentiment_label,
                                     tb_sentiment, nltk_sentiment, row_avg)
from graph_utils.graph import (plot_word_freq, two_dim_tf_viz,
                               time_series_line_viz)

# read trump .csv files & union them into a single DataFrame: trump_df
trump_df = union_csv(
    csv_path=r'C:\Users\billy\presidential election\nytimes api\article search\trump_politics_all',
    glob_pattern='Donald Trump_*.csv'
)

print('DataFrame Sample')
print(trump_df.head())
print()
print('DataFrame Shape: {}'.format(trump_df.shape))
print()

# create 'text' feature
# drop null rows from the 'abstract' column
# return preprocessed DataFrame with 'headline_main','web_url','text','word_count','pub_date' columns
trump_df = preprocess_df(
    df=trump_df,
    new_col='text',
    con_col1='headline_main',
    con_col2='abstract',
    subset_list=['abstract'],
    filter_col_list=['headline_main', 'web_url', 'text', 'word_count', 'pub_date']
)

# filter out rows that don't mention Trump
trump_df = trump_df[trump_df['text'].str.contains('trump', case=False)]
print('Preprocessed DataFrame!')
print()
print('New DataFrame Shape: {}'.format(trump_df.shape))
print()

# generate date_features for trump_df
trump_df = date_feats(trump_df, 'pub_date')

# generate TextBlob sentiment and subjectivity scores for each row of 'text' in trump_df: tb_sent_scores
# create a new DataFrame filled with scores: tb_sentiment_df
# join sentiment scores to original trump_df DataFrame
tb_sent_scores = [tb_sentiment(text=row) for row in trump_df['text']]
tb_sentiment_df = pd.DataFrame(tb_sent_scores)
trump_df = trump_df.join(tb_sentiment_df)
# delete unused variables
del tb_sent_scores, tb_sentiment_df
print('Generated TextBlob Sentiment Scores!')
print()
print(trump_df[trump_df['polarity'] == trump_df['polarity'].max()]['text'].values)
print()

# generate Vader sentiment scores for each row of 'text' in trump_df: nltk_sent_scores
# create a new DataFrame filled with scores: nltk_sentiment_df
# join sentiment scores to original trump_df DataFrame
nltk_sent_scores = [nltk_sentiment(text=row) for row in trump_df['text']]
nltk_sentiment_df = pd.DataFrame(nltk_sent_scores)
trump_df = trump_df.join(nltk_sentiment_df)
# delete unused variables
del nltk_sent_scores, nltk_sentiment_df
print('Generated Vader Sentiment Scores!')
print()
print(trump_df[trump_df['compound'] == trump_df['compound'].max()]['text'].values)
print()

# compute the average of the Vader and TextBlob sentiment scores: trump_df['sentiment']
trump_df = row_avg(trump_df, 'compound', 'polarity', 'sentiment')
print('Computed Mean Sentiment Scores!')
print()
print('Positive Sentiment')
print(trump_df[trump_df['sentiment'] == trump_df['sentiment'].max()]['text'].values)
print()
print('Negative Sentiment')
print(trump_df[trump_df['sentiment'] == trump_df['sentiment'].min()]['text'].values)
print()
print('Neutral Sentiment')
print(trump_df[trump_df['sentiment'] == 0]['text'].values)
print()

# compute the labels for modelling:
article_df = sentiment_label(
    df=trump_df,
    col_for_label='sentiment',
    label_col='sentiment_label'
)
print('Computed labels for Modelling')
print()
print(trump_df['sentiment_label'].unique())
print()

# filter out rows that have np.nan values
trump_df = trump_df[trump_df['sentiment_label'] != 'nan']
print('Not Null DataFrame Metadata: {}'.format(trump_df.info()))
print()
print(trump_df['sentiment_label'].value_counts())
print()

# plot word frequencies
plot_word_freq(
    pd_series=trump_df[trump_df['sentiment_label'] == 'positive']['text'],
    plot_title='Trump BOW FREQUENCY',
    n=30
)
print('saved Bar Graph .png!')
print()

# plot tfidf Scatter plot
two_dim_tf_viz(
    df=trump_df,
    pd_series='text',
    pd_color_series='sentiment_label',
    pd_hover_series='headline_main',
    max_features=800,
    plot_title='N.Y. Times Trump Articles Sentiment Clusters'
)
print('saved Scatter Plot .png!')
print()

# plot Time Series Line plot
time_series_line_viz(
    df=trump_df,
    date_index='pub_date',
    pd_series='sentiment',
    plot_title='N.Y. Times Trump Articles Avg. Daily Sentiment'
)
print('saved Line Plot .png!')
print()
