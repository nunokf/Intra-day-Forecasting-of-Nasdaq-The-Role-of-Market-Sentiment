from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Initialize the API client with your API key
api_key = 'fcd8f6df1ed34bf0baf86b85bf4fb677'
newsapi = NewsApiClient(api_key=api_key)

# Load FinBERT
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to clean and preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Function to analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach()
    sentiment_index = torch.argmax(scores, dim=1).item()  # Get the sentiment index
    confidence_score = torch.max(scores).item()  # Get the confidence score

    # Map the sentiment index to defined scores
    if sentiment_index == 2:  # Positive (fear)
        return 1, confidence_score
    elif sentiment_index == 1:  # Neutral
        return 0, confidence_score
    else:  # Negative (no fear)
        return -1, confidence_score

def finbert(search = 'US stock market OR US economy OR S&P 500 OR Dow Jones OR Nasdaq'):
    #sectors = ["energy", "technology", "healthcare"]
    # Fetch top finance news articles
    finance_news = newsapi.get_everything(
        q=search,
        language='en',
        sort_by='relevancy',
        page_size=100,
        from_param='2024-10-02',
        to='2024-11-02'
    )

    import numpy as np

    # Initialize lists for sentiment scores and article counts
    sentiments = {}
    article_counts = {}

    # Analyze sentiment for each article
    for article in finance_news['articles']:
        published_at = article['publishedAt']
        date_str = published_at.split('T')[0]  # Extract the date (YYYY-MM-DD)

        # Preprocess title and description as a single string
        title = preprocess_text(article['title'])
        description = preprocess_text(article['description']) if article['description'] else ""
        combined_text = f"{title} {description}"  # Concatenate title and description

        # Analyze sentiment of the combined text
        sentiment, _ = analyze_sentiment(combined_text)

        # Aggregate sentiment scores and article counts
        if date_str not in sentiments:
            sentiments[date_str] = 0
            article_counts[date_str] = 0

        sentiments[date_str] += sentiment
        article_counts[date_str] += 1  # Count the article

    # Convert sentiments and counts to lists for normalization
    sentiment_sums = np.array(list(sentiments.values()))
    counts = np.array(list(article_counts.values()))

    # Normalize the sentiment sums by the article counts
    normalized_sentiment_sums = sentiment_sums / counts

    return normalized_sentiment_sums, sentiments


def dataframe_plotting(sentiments,normalized_sentiment_sums):
    # Convert date strings to datetime objects
    dates = list(sentiments.keys())
    dates_dt = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Create a continuous date range for the x-axis
    full_date_range = pd.date_range(start=min(dates_dt), end=max(dates_dt), freq='D')

    # Create a full sentiment sums array with zeros for missing dates
    full_sentiment_sums = np.zeros(len(full_date_range))

    # Map existing normalized sentiment sums to the corresponding dates
    for i, date in enumerate(full_date_range):
        if date.strftime('%Y-%m-%d') in sentiments:
            # Get the index of the current date in the dates list
            index = dates.index(date.strftime('%Y-%m-%d'))
            full_sentiment_sums[i] = normalized_sentiment_sums[index]

    # Create a DataFrame for plotting
    df_full = pd.DataFrame({
        'Date': full_date_range,
        'Sentiment': full_sentiment_sums
    })
    df_full.set_index('Date', inplace=True)

    return df_full


# DTW function
def fill_dtw_cost_matrix(s1, s2):
    l_s_1, l_s_2 = len(s1), len(s2)
    cost_matrix = np.zeros((l_s_1 + 1, l_s_2 + 1))
    for i in range(l_s_1 + 1):
        for j in range(l_s_2 + 1):
            cost_matrix[i, j] = np.inf
    cost_matrix[0, 0] = 0

    for i in range(1, l_s_1 + 1):
        for j in range(1, l_s_2 + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            prev_min = np.min([cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]])
            cost_matrix[i, j] = cost + prev_min

    # Backtrack to find the path
    path = []
    i, j = l_s_1, l_s_2
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))  # Store the indices in the path
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if cost_matrix[i - 1, j] < cost_matrix[i, j - 1] and cost_matrix[i - 1, j] < cost_matrix[i - 1, j - 1]:
                i -= 1
            elif cost_matrix[i, j - 1] < cost_matrix[i - 1, j] and cost_matrix[i, j - 1] < cost_matrix[i - 1, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1

    path.reverse()  # Reverse to get the path from start to end
    return cost_matrix, path

# Normalization function
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())
