import streamlit as st
import plotly.express as px
from datetime import date
import pandas as pd
from utils import *





# Contents of ~/my_app/pages/page_2.py
import streamlit as st

st.markdown("# Page 2")
st.sidebar.markdown("# Page 2")

# Contents of ~/my_app/pages/page_3.py
import streamlit as st

st.markdown("# Page 3")
st.sidebar.markdown("# Page 3")





df_tweet, df_reply = load_data()

# Convert date column to datetime format
df_tweet['date_only'] = pd.to_datetime(df_tweet['date_only'])
df_reply['date_only'] = pd.to_datetime(df_reply['date_only'])

# Create a sidebar with a date input
start_date = st.sidebar.date_input("Start date", value=date(2022, 1, 8))
end_date = st.sidebar.date_input("End date", value=date(2022, 1, 14))

# Convert the date inputs to datetime objects
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())

# Filter the data based on the date input
(tweet_filtered, tweet_per_date, tweet_per_hour, tweet_filtered_previous, tweet_per_date_previous, 
            tweet_per_hour_previous,popularity_per_date, popularity_per_hour, popularity_per_date_previous,
            popularity_per_hour_previous, reply_filtered, reply_per_date, reply_per_hour, reply_filtered_previous, 
            reply_per_date_previous, reply_per_hour_previous, top_popular_tweets, top_popular_replies, 
            top_controversial_tweets, top_controversial_replies) = filtering_wrap(df_tweet, df_reply, start_date, end_date)


top_popular_tweets_list = top_popular_tweets["id"].to_list()
top_popular_replies_list = top_popular_replies["reply_id"].to_list()
top_popular_replies_name = top_popular_replies["user_name"].to_list()
#concatenated_df = add_date_column_and_concatenate(tweet_per_date, tweet_per_date_previous)

# Create a two-column layout with the charts

col1, col2 = st.columns(2)

with col1:
   st.header("#1 Tweet")
   t = Tweet(f"https://twitter.com/detikcom/status/{top_popular_tweets_list[0]}").component()

with col2:
   st.header("#2 Tweet")
   t = Tweet(f"https://twitter.com/detikcom/status/{top_popular_tweets_list[1]}").component()


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

col3, col4 = st.columns(2)
with col3:
   st.header("#1 Reply")
   t = TweetReply(f"https://twitter.com/{top_popular_replies_name[0]}/status/{top_popular_replies_list[0]}").component()

with col4:
   st.header("#2 Reply")
   t = TweetReply(f"https://twitter.com/{top_popular_replies_name[1]}/status/{top_popular_replies_list[1]}").component()

