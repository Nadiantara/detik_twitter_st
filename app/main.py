import streamlit as st
import plotly.express as px
from datetime import date
import pandas as pd
from utils import *




st.set_page_config(
    page_title="Twitter Dashboard Overview",
    page_icon="👋",
)

# Create a sidebar with a date input
start_date = st.sidebar.date_input("Start date", value=date(2022, 1, 8))
end_date = st.sidebar.date_input("End date", value=date(2022, 1, 14))

df_tweet, df_reply = load_data()

# Convert date column to datetime format
df_tweet['date_only'] = pd.to_datetime(df_tweet['date_only'])
df_reply['date_only'] = pd.to_datetime(df_reply['date_only'])


# Convert the date inputs to datetime objects
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())

# Filter the data based on the date input
(tweet_filtered, tweet_per_date, tweet_per_hour, tweet_filtered_previous, tweet_per_date_previous, 
            tweet_per_hour_previous,popularity_per_date, popularity_per_hour, popularity_per_date_previous,
            popularity_per_hour_previous, reply_filtered, reply_per_date, reply_per_hour, reply_filtered_previous, 
            reply_per_date_previous, reply_per_hour_previous, top_popular_tweets, top_popular_replies, 
            top_controversial_tweets, top_controversial_replies) = filtering_wrap(df_tweet, df_reply, start_date, end_date)

#concatenated_df = add_date_column_and_concatenate(tweet_per_date, tweet_per_date_previous)




lineplot1 = daily_tweet(tweet_per_date)
lineplot2 = daily_popularity(popularity_per_date)
lineplot3 = hourly_popularity(popularity_per_hour)
lineplot4 = daily_engagement(reply_per_date)
lineplot5 = hourly_engagement(reply_per_hour)


st.plotly_chart(lineplot1)
st.plotly_chart(lineplot2)
st.plotly_chart(lineplot3)
st.plotly_chart(lineplot4)
st.plotly_chart(lineplot5)



