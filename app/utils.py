import pandas as pd

def load_data():
    detik_tweet = pd.read_csv("/data/all_cleaned_detik_tweet_merged.csv")
    detik_reply = pd.read_csv("data/all_cleaned_reply_tweet_merged.csv")
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
    previous_start_date = start_date - pd.Timedelta(days=(end_date - start_date).days)
    previous_end_date = start_date - pd.Timedelta(days=1)

    # Filter by date range
    filtered_df = df[(df[date_col] >= previous_start_date) & (df[date_col] <= end_date)]

    # Sort by date ascending
    sorted_df = filtered_df.sort_values('date_col', ascending=True)

    return sorted_df
