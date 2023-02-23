import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
import requests
import streamlit.components.v1 as components

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DATA_PATH = os.path.dirname(currentdir)

@st.cache_data(ttl=300)
def load_data():
    detik_tweet = pd.read_csv(f"{DATA_PATH}/data/all_cleaned_detik_tweet_merged.csv")
    detik_reply = pd.read_csv(f"{DATA_PATH}/data/all_cleaned_reply_tweet_merged.csv")
    return detik_tweet, detik_reply


def filter_date_range(df, date_col, start_date, end_date):
    """Filter a pandas DataFrame based on a date range.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        date_col (str): The name of the date column in the DataFrame.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter DataFrame based on date range
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    filtered_df = df.loc[mask]

    return filtered_df.sort_values(by=date_col, ascending=True)


def filter_by_date_with_previous_period(df,date_col, start_date, end_date):
    """Filter a DataFrame by a custom start date and end date, with a previous period included.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        start_date (str): The start date in yyyy-mm-dd format.
        end_date (str): The end date in yyyy-mm-dd format.

    Returns:
        pandas.DataFrame: A filtered DataFrame with previous period included, sorted by date ascending.
    """
    # Convert start and end dates to pandas datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Calculate start and end dates for previous period
    previous_start_date = start_date - timedelta(days=(end_date - start_date).days+1)
    previous_end_date = end_date - timedelta(days=(end_date - start_date).days+1)

    # Filter by date range
    filtered_df = df[(df[date_col] >= previous_start_date) & (df[date_col] <= previous_end_date)]

    # Sort by date ascending
    sorted_df = filtered_df.sort_values(date_col, ascending=True)

    return sorted_df


def count_by_group(df, group_col, count_col):
    """Group by a column and count the occurrences of each value.

    Args:
        df (pandas.DataFrame): The DataFrame to group by.
        group_col (str): The name of the column to group by.
        count_col (str): The name of the column to count.

    Returns:
        pandas.DataFrame: A DataFrame with the count of occurrences for each group.
    """
    counts = df.groupby(group_col)[count_col].count()
        # Convert the resulting Series to a DataFrame with a descriptive column name
    result = pd.DataFrame({'total_count': counts})

    # Reset the index to include the grouping column
    result.reset_index(inplace=True)

    return result


def sum_by_group(df, group_col, sum_col):
    """Group by a column and sum the values in another column.

    Args:
        df (pandas.DataFrame): The DataFrame to group by.
        group_col (str): The name of the column to group by.
        sum_col (str): The name of the column to sum.

    Returns:
        pandas.DataFrame: A DataFrame with the sum of values for each group.
    """
    
    # Group the DataFrame by the specified column and sum the other column
    grouped = df.groupby(group_col)[sum_col].sum()

    # Convert the resulting Series to a DataFrame with a descriptive column name
    result = pd.DataFrame({'total_value': grouped})

    # Reset the index to include the grouping column
    result.reset_index(inplace=True)

    # Return the result
    return result

@st.cache_data(ttl=300)
def filtering_wrap(df_tweet, df_reply, start_date, end_date):
    # detik's tweet
    tweet_filtered = filter_date_range(df_tweet, "date_only", start_date, end_date)
    tweet_per_date = count_by_group(tweet_filtered, "date_only", "id")
    tweet_per_hour = count_by_group(tweet_filtered, "hour", "id")
    tweet_filtered_previous = filter_by_date_with_previous_period(df_tweet, "date_only", start_date, end_date)
    tweet_per_date_previous = count_by_group(tweet_filtered_previous, "date_only", "id")
    tweet_per_hour_previous = count_by_group(tweet_filtered_previous, "hour", "id")
    
    #popularity score
    popularity_per_date = sum_by_group(tweet_filtered, "date_only", "popularity_score")
    popularity_per_hour = sum_by_group(tweet_filtered, "hour", "popularity_score")
    popularity_per_date_previous = sum_by_group(tweet_filtered_previous, "date_only", "popularity_score")
    popularity_per_hour_previous = sum_by_group(tweet_filtered_previous, "hour", "popularity_score")
    
    #controversiality score
    controversiality_per_date = sum_by_group(tweet_filtered, "date_only", "controversiality_score")
    controversiality_per_hour = sum_by_group(tweet_filtered, "hour", "controversiality_score")
    controversiality_per_date_previous = sum_by_group(tweet_filtered_previous, "date_only", "controversiality_score")
    controversiality_per_hour_previous = sum_by_group(tweet_filtered_previous, "hour", "controversiality_score")

    # reply
    reply_filtered = filter_date_range(df_reply, "date_only", start_date, end_date)
    reply_per_date = count_by_group(reply_filtered, "date_only", "reply_id")
    reply_per_hour = count_by_group(reply_filtered, "hour", "reply_id")
    reply_filtered_previous = filter_by_date_with_previous_period(df_reply, "date_only", start_date, end_date)
    reply_per_date_previous = count_by_group(reply_filtered_previous, "date_only", "reply_id")
    reply_per_hour_previous = count_by_group(reply_filtered_previous, "hour", "reply_id")

    # popular and controversial
    top_popular_tweets = tweet_filtered.sort_values(by="popularity_score", ascending=False).head(4)
    top_popular_replies = reply_filtered.sort_values(by="popularity_score", ascending=False).head(4)
    top_controversial_tweets = tweet_filtered.sort_values(by="controversiality_score", ascending=False).head(4)
    top_controversial_replies = reply_filtered.sort_values(by="controversiality_score", ascending=False).head(4)
    
    result_dict = {
        "tweet_per_date": tweet_per_date,
        "tweet_per_hour": tweet_per_hour,
        "tweet_per_date_previous": tweet_per_date_previous,
        "tweet_per_hour_previous": tweet_per_hour_previous,
        "popularity_per_date": popularity_per_date,
        "popularity_per_hour": popularity_per_hour,
        "popularity_per_date_previous": popularity_per_date_previous,
        "popularity_per_hour_previous": popularity_per_hour_previous,
        "controversiality_per_date": controversiality_per_date,
        "controversiality_per_hour": controversiality_per_hour,
        "controversiality_per_date_previous": controversiality_per_date_previous,
        "controversiality_per_hour_previous": controversiality_per_hour_previous,
        "reply_per_date": reply_per_date,
        "reply_per_hour": reply_per_hour,
        "reply_per_date_previous": reply_per_date_previous,
        "reply_per_hour_previous": reply_per_hour_previous,
        "top_popular_tweets": top_popular_tweets,
        "top_popular_replies": top_popular_replies,
        "top_controversial_tweets": top_controversial_tweets,
        "top_controversial_replies": top_controversial_replies,
    }

    return result_dict
    
    return (tweet_filtered, tweet_per_date, tweet_per_hour, tweet_filtered_previous, tweet_per_date_previous, 
            tweet_per_hour_previous,popularity_per_date, popularity_per_hour, popularity_per_date_previous, popularity_per_hour_previous, reply_filtered, reply_per_date, reply_per_hour, reply_filtered_previous, 
            reply_per_date_previous, reply_per_hour_previous, top_popular_tweets, top_popular_replies, 
            top_controversial_tweets, top_controversial_replies)






def add_date_column_and_concatenate(A, B):
    """
    Adds a new period column to two Pandas DataFrames and concatenates them.

    Args:
        A (pandas.DataFrame): The first DataFrame to concatenate.
        B (pandas.DataFrame): The second DataFrame to concatenate.

    Returns:
        pandas.DataFrame: The concatenated DataFrame.

    """
    # Add new column to A and B
    A['period'] = 'This Period'
    B['period'] = 'Previous Period'
    B["date_only"] = A["date_only"]

    # Concatenate A and B
    concatenated_df = pd.concat([A, B], ignore_index=True)

    return concatenated_df 

def daily_tweet(df):
    df = df.rename(columns={"date_only": "date", "total_count": "total"})
    fig = px.line(df, x='date', y='total', title='Total Tweet Published per Day')
    return fig

def daily_popularity(df):
    df = df.rename(columns={"date_only": "date", "total_value": "total"})
    fig = px.line(df, x='date', y='total', title=' Tweet Popularity per Day')
    return fig

def hourly_popularity(df):
    df = df.rename(columns={"total_value": "total"})
    fig = px.line(df, x='hour', y='total', title=' Tweet Popularity per Hour')
    return fig

def daily_engagement(df):
    df = df.rename(columns={"date_only": "date", "total_count": "total"})
    fig = px.line(df, x='date', y='total', title='Users Replies per Day')
    return fig

def hourly_engagement(df):
    df = df.rename(columns={"hour": "hour", "total_count": "total"})
    fig = px.line(df, x='hour', y='total', title='Users Replies per Hour')
    return fig


def plot_metrics_by_date(df, color_1="blue", color_2="lightblue", y_title = "tweets published" ):
    # Rename the columns in the dataframe
    df = df.rename(columns={"date_only": "date", "total_count": y_title})

    # Create the line chart
    fig = px.line(df, x='date', y='tweets published', color='period', symbol="period",
                  color_discrete_map={"This Period": color_1, "Previous Period": color_2})

    # Update the chart layout
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white'
    )

    fig.show()


def fill_missing_rows(A, B):
    # Get a set of unique dates in A
    A_dates = set(A.iloc[:, 0].tolist())

    # Create a new dataframe B' with the same dates as A
    B_cols = B.columns.tolist()
    B_prime = pd.DataFrame(columns=B_cols)

    # Fill in the values from B where dates match
    if not B.empty:
        for i in range(B.shape[0]):
            date = B.iloc[i, 0]
            val = B.iloc[i, 1]
            B_prime = B_prime.append(pd.DataFrame([[date, val]], columns=B_cols), ignore_index=True)

    # Fill in any missing dates with 0
    for date in A_dates:
        if date not in B_prime.iloc[:, 0].tolist():
            B_prime = B_prime.append(pd.DataFrame([[date, 0]], columns=B_cols), ignore_index=True)

    # Sort the rows by date
    B_prime = B_prime.sort_values(by=B_cols[0])

    return B_prime


def check_same_rows(A, B):
    return A.shape[0] == B.shape[0]

def compute_now_previous(A,B):
    if check_same_rows(A,B):
        concatenated_df = add_date_column_and_concatenate(A,B)
    else:
        B = fill_missing_rows(A,B)
        concatenated_df = add_date_column_and_concatenate(A,B)
    return concatenated_df



def calc_period_percent_diff(df):
    # Separate the dataframe into "This Period" and "Previous Period"
    this_period = df[df['period'] == 'This Period']
    prev_period = df[df['period'] == 'Previous Period']

    # Calculate the sum of values for each period
    this_period_sum = this_period.iloc[:, 1].sum()
    prev_period_sum = prev_period.iloc[:, 1].sum()

    # Calculate the percentage difference between the periods
    if prev_period_sum == 0:
        percent_diff="Not Available"    
    percent_diff = ((this_period_sum - prev_period_sum) / prev_period_sum) * 100
    
    return this_period_sum, percent_diff


class Tweet(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = "https://publish.twitter.com/oembed?url={}".format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=500)
    
    
class TweetReply(Tweet):
    def component(self):
        return components.html(self.text, height=800)