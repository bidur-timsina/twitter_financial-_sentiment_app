# Financial Twitter Sentiment Analyzer 📈

## Overview

The **Financial Twitter Sentiment Analyzer** is a Streamlit-based web application that helps users analyze the sentiment of financial discussions on Twitter. The app fetches tweets related to specific financial topics (stocks, crypto, market trends, etc.) and performs sentiment analysis using natural language processing (NLP) techniques. The sentiment scores are enhanced with financial terminology, allowing users to gauge the market sentiment for a given asset or topic.

This tool provides visualizations like sentiment distribution, engagement metrics, word clouds, and sentiment over time.

## Features

- **Twitter API Integration**: Fetch tweets using the Twitter API v2 with rate limit handling.
- **Sentiment Analysis**: Sentiment is analyzed with VADER (Valence Aware Dictionary and sEntiment Reasoner) and financial term weighting.
- **Visualizations**: Includes:
  - Sentiment distribution (Pie chart)
  - Sentiment score distribution (Histogram)
  - Engagement analysis (Scatter plot with likes and retweets)
  - Word cloud of financial discussions
  - Sentiment trend over time
- **Caching**: Tweet data is cached to avoid excessive API calls and enhance performance.
- **API Token**: Secure input for the Twitter API Bearer Token.
- **Custom Search**: Allows users to search for specific financial terms or preset topics (stocks, crypto, indexes).

## Requirements

- Python 3.7 or higher
- Streamlit
- Tweepy
- VADER Sentiment Analyzer
- Pandas
- Plotly
- Matplotlib
- WordCloud

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clonehttps://github.com/bidur-timsina/twitter_financial-_sentiment_app
   cd financial-twitter-sentiment-analyzer
