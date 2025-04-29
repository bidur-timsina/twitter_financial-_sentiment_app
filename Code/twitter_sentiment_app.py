import streamlit as st
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import time
from wordcloud import WordCloud

st.set_page_config(
    page_title="Financial Twitter Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0E76A8;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2E8B57;
        text-align: center;
    }
    .stButton>button {
        background-color: #0E76A8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Financial Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Analyze market sentiment for stocks, crypto, and financial topics</p>", unsafe_allow_html=True)

if 'bearer_token' not in st.session_state:
    st.session_state.bearer_token = ""
    
with st.sidebar:
    st.title("API Configuration")
    
# Twitter Bearer Token input with secure password input
    bearer_token = st.text_input(
        "Twitter API Bearer Token", 
        type="password", 
        value=st.session_state.bearer_token,
        help="Enter your Twitter API Bearer Token for authentication"
    )
    
    if bearer_token:
        st.session_state.bearer_token = bearer_token
    
    if st.session_state.bearer_token:
        st.success("✅ Bearer token set")
    else:
        st.warning("⚠️ Please enter your Twitter API Bearer Token")
    
    if st.session_state.bearer_token and st.button("Clear Token"):
        st.session_state.bearer_token = ""
        st.experimental_rerun()

financial_terms = {
    "bullish": 0.7,
    "bearish": -0.7,
    "uptrend": 0.6,
    "downtrend": -0.6,
    "buy": 0.5,
    "sell": -0.5,
    "long": 0.4,
    "short": -0.4,
    "calls": 0.5,
    "puts": -0.5,
    "rally": 0.6,
    "crash": -0.8,
    "moon": 0.8,
    "dip": -0.3,
    "recession": -0.7,
    "growth": 0.6,
    "profit": 0.7,
    "loss": -0.7,
    "earnings": 0.2,
    "miss": -0.6,
    "beat": 0.6,
    "dividend": 0.5,
    "upgrade": 0.7,
    "downgrade": -0.7,
    "outperform": 0.7,
    "underperform": -0.7
}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def clean_tweet(tweet):
    """Clean the tweet text by removing URLs, mentions, hashtags, and special characters"""
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet.lower().strip()

def get_financial_sentiment_score(text):
    """Get sentiment score with financial term weighting"""
    base_score = analyzer.polarity_scores(text)
    compound = base_score['compound']
    text_lower = text.lower()
    for term, weight in financial_terms.items():
        if term in text_lower:
            compound = compound + (weight * 0.1)  # Apply 10% weight
    
    return max(-1.0, min(1.0, compound))

def get_sentiment_label(score):
    """Convert sentiment score to a categorical label with financial thresholds"""
    if score > 0.1:
        return "Bullish"
    elif score < -0.1:
        return "Bearish"
    else:
        return "Neutral"

def get_sentiment_color(sentiment):
    """Return color based on financial sentiment"""
    colors = {"Bullish": "#2E8B57", "Neutral": "#708090", "Bearish": "#CD5C5C"}
    return colors.get(sentiment, "#708090")

def fetch_financial_tweets(query, count, bearer_token):
    """Fetch tweets using Twitter API v2 with rate limit handling"""
    if not bearer_token:
        st.error("Please enter your Twitter API Bearer Token in the sidebar")
        return None
        
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        financial_query = f"{query} (stock OR market OR price OR invest OR trading OR finance OR earnings) lang:en -is:retweet"
        
        with st.spinner(f"Fetching financial tweets about '{query}'..."):
            batch_size = min(count, 25)  
            max_attempts = 3
            wait_time = 2  
            
            for attempt in range(max_attempts):
                try:
                    response = client.search_recent_tweets(
                        query=financial_query, 
                        max_results=batch_size, 
                        tweet_fields=['created_at', 'public_metrics', 'context_annotations']
                    )
                    break
                except tweepy.TooManyRequests:
                    if attempt < max_attempts - 1:  
                        st.warning(f"Rate limit hit. Waiting {wait_time} seconds... (Attempt {attempt+1}/{max_attempts})")
                        time.sleep(wait_time)
                        wait_time *= 2  
                    else:
                        st.error("Twitter API rate limit exceeded. Please try again later with fewer tweets or wait a few minutes.")
                        return None
        
        if not response.data:
            return None
            
        tweets = []
        for tweet in response.data:
            tweets.append({
                "text": tweet.text,
                "created_at": tweet.created_at,
                "likes": tweet.public_metrics['like_count'] if hasattr(tweet, 'public_metrics') else 0,
                "retweets": tweet.public_metrics['retweet_count'] if hasattr(tweet, 'public_metrics') else 0
            })
            
        return pd.DataFrame(tweets)
    except tweepy.Unauthorized:
        st.error("Authentication failed. Please check your Twitter API Bearer Token.")
        return None
    except tweepy.BadRequest as e:
        st.error(f"Bad request: {str(e)}. Please check your search query.")
        return None
    except tweepy.TooManyRequests:
        st.error("Twitter API rate limit exceeded. Please try again later with fewer tweets or wait a few minutes.")
        return None
    except Exception as e:
        st.error(f"Error fetching tweets: {str(e)}")
        return None


analyzer = SentimentIntensityAnalyzer()


if 'tweet_cache' not in st.session_state:
    st.session_state.tweet_cache = {}
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = datetime.now()


def get_tweets(query, count, bearer_token, enable_cache=True, cache_ttl=30):
    """Get tweets from cache or fetch new ones if needed"""
    cache_key = f"{query}_{count}"
    current_time = datetime.now()
    

    if enable_cache and cache_key in st.session_state.tweet_cache:
        cache_data, cache_time = st.session_state.tweet_cache[cache_key]
        time_diff = (current_time - cache_time).total_seconds() / 60
        
        if time_diff < cache_ttl:
            return cache_data, True
    

    df = fetch_financial_tweets(query, count, bearer_token)
    

    if df is not None and not df.empty and enable_cache:
        st.session_state.tweet_cache[cache_key] = (df, current_time)
        st.session_state.last_query_time = current_time
    
    return df, False 

# Sidebar for inputs
with st.sidebar:
    st.title("Analysis Settings")
    

    preset_topics = ["AAPL", "MSFT", "TSLA", "BTC", "ETH", "S&P500", "DJIA", "inflation", "fed rates", "earnings"]
    
    st.subheader("Search Parameters")
    query_type = st.radio("Search Type", ["Preset Topics", "Custom Search"])
    
    if query_type == "Preset Topics":
        query = st.selectbox("Select Financial Topic", preset_topics)
    else:
        query = st.text_input("Custom Search Term", "AAPL")
    

    if len(query) <= 5 and query.isalpha() and query_type != "Preset Topics":
        cashtag = st.checkbox("Add $ symbol (cashtag)", True)
        if cashtag:
            query = f"${query}"
    
    tweet_count = st.slider("Number of tweets to analyze", 5, 50, 15, 
                           help="⚠️ Higher values may trigger Twitter API rate limits. Start with fewer tweets.")
    
    with st.expander("Advanced API Settings"):
        st.caption("These settings help avoid Twitter API rate limits")
        enable_cache = st.checkbox("Enable tweet caching", True, 
                                 help="Cache tweets to reduce API calls")
        cache_ttl = st.slider("Cache duration (minutes)", 5, 120, 30,
                            help="How long to keep cached results")
        
        if 'last_query_time' in st.session_state and enable_cache:
            last_time = st.session_state.get('last_query_time', datetime.now())
            time_diff = (datetime.now() - last_time).total_seconds() / 60
            if time_diff < cache_ttl:
                st.info(f"Cache active for {cache_ttl-int(time_diff)} more minutes")
    
    time_period = st.selectbox("Time Period", ["Recent Tweets", "Last 24 Hours", "Last 7 Days"])
    
    st.subheader("Display Options")
    display_raw_tweets = st.checkbox("Show raw tweets", True)
    display_wordcloud = st.checkbox("Show word cloud", True)
    display_time_series = st.checkbox("Show sentiment over time", True)
    display_engagement = st.checkbox("Show engagement metrics", True)
    
    analyze_button = st.button("Analyze Market Sentiment", type="primary")

# Main app logic
if analyze_button:

    if not st.session_state.bearer_token:
        st.error("Please enter your Twitter API Bearer Token in the sidebar")
    else:
        # Fetch and process tweets (with caching)
        with st.status("Processing", expanded=True) as status:
            status.update(label="Checking for cached data...")
            df, from_cache = get_tweets(
                query, 
                tweet_count, 
                st.session_state.bearer_token,
                enable_cache=st.session_state.get('enable_cache', True),
                cache_ttl=st.session_state.get('cache_ttl', 30)
            )
            
            if from_cache:
                status.update(label="✅ Using cached tweets to avoid rate limits")
                
        if df is not None and not df.empty:
            # Process the tweets for sentiment analysis
            df['clean_text'] = df['text'].apply(clean_tweet)
            df['sentiment_score'] = df['clean_text'].apply(get_financial_sentiment_score)
            df['sentiment'] = df['sentiment_score'].apply(get_sentiment_label)
            df['color'] = df['sentiment'].apply(get_sentiment_color)
            
            avg_sentiment = df['sentiment_score'].mean()
            sentiment_status = "BULLISH" if avg_sentiment > 0.1 else "BEARISH" if avg_sentiment < -0.1 else "NEUTRAL"
            sentiment_color = "#2E8B57" if avg_sentiment > 0.1 else "#CD5C5C" if avg_sentiment < -0.1 else "#708090"
            
            st.markdown(f"""
            <div style="text-align:center; padding:10px; background-color:{sentiment_color}; color:white; border-radius:5px; margin-bottom:20px;">
                <h2>Market Sentiment for {query}: {sentiment_status}</h2>
                <p>Average Sentiment Score: {avg_sentiment:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
# Display summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bullish_count = (df['sentiment'] == 'Bullish').sum()
                st.metric("Bullish Tweets", bullish_count, f"{bullish_count/len(df):.1%}")
                
            with col2:
                neutral_count = (df['sentiment'] == 'Neutral').sum()
                st.metric("Neutral Tweets", neutral_count, f"{neutral_count/len(df):.1%}")
                
            with col3:
                bearish_count = (df['sentiment'] == 'Bearish').sum()
                st.metric("Bearish Tweets", bearish_count, f"{bearish_count/len(df):.1%}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution pie chart
                fig = px.pie(
                    df, 
                    names='sentiment',
                    color='sentiment',
                    color_discrete_map={
                        "Bullish": "#2E8B57", 
                        "Neutral": "#708090", 
                        "Bearish": "#CD5C5C"
                    },
                    title=f"Sentiment Distribution for '{query}'"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    df, 
                    x='sentiment_score',
                    color='sentiment',
                    color_discrete_map={
                        "Bullish": "#2E8B57", 
                        "Neutral": "#708090", 
                        "Bearish": "#CD5C5C"
                    },
                    title="Distribution of Sentiment Scores",
                    labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Tweets'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if display_engagement:
                st.subheader("Engagement Analysis")
                
                fig = px.scatter(
                    df,
                    x='sentiment_score',
                    y='likes',
                    size='retweets',
                    color='sentiment',
                    color_discrete_map={
                        "Bullish": "#2E8B57", 
                        "Neutral": "#708090", 
                        "Bearish": "#CD5C5C"
                    },
                    title=f"Engagement vs Sentiment for '{query}'",
                    labels={
                        'sentiment_score': 'Sentiment Score', 
                        'likes': 'Likes', 
                        'retweets': 'Retweets'
                    },
                    hover_data=['text']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if display_wordcloud:
                st.subheader("Word Cloud of Financial Discussions")
                
                all_words = ' '.join(df['clean_text'])
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    contour_width=1,
                    contour_color='steelblue',
                    collocations=False,
                    colormap='YlGnBu'
                ).generate(all_words)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            if display_time_series:
                st.subheader("Sentiment Trend Over Time")
                
                # Convert to datetime and sort
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at')
                
                # Calculate rolling average of sentiment scores
                df['rolling_sentiment'] = df['sentiment_score'].rolling(window=5, min_periods=1).mean()
                
                # Create time series chart
                fig = px.line(
                    df, 
                    x='created_at', 
                    y='rolling_sentiment',
                    title=f"Sentiment Trend for '{query}' (5-tweet rolling average)",
                    labels={'rolling_sentiment': 'Sentiment Score', 'created_at': 'Time'}
                )
                
                # Add horizontal lines for bullish/bearish thresholds
                fig.add_hline(y=0.1, line_dash="dash", line_color="#2E8B57", annotation_text="Bullish Threshold")
                fig.add_hline(y=-0.1, line_dash="dash", line_color="#CD5C5C", annotation_text="Bearish Threshold")
                
                st.plotly_chart(fig, use_container_width=True)
            
            if display_raw_tweets:
                st.subheader("Analyzed Financial Tweets")
                
                display_df = df[['text', 'sentiment', 'sentiment_score', 'created_at', 'likes', 'retweets']].rename(columns={
                    'text': 'Tweet',
                    'sentiment': 'Sentiment',
                    'sentiment_score': 'Score',
                    'created_at': 'Posted At',
                    'likes': 'Likes',
                    'retweets': 'Retweets'
                })
                
                def highlight_sentiment(val):
                    if val == 'Bullish':
                        return 'background-color: rgba(46, 139, 87, 0.2)'
                    elif val == 'Bearish':
                        return 'background-color: rgba(205, 92, 92, 0.2)'
                    return ''
                
                styled_df = display_df.style.applymap(highlight_sentiment, subset=['Sentiment'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Add download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    f"financial_sentiment_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="download-csv"
                )
        else:
            st.error("No financial tweets found matching your criteria or API request failed.")
else:
    # Display welcome message and instructions
    if not st.session_state.bearer_token:
        st.warning("⚠️ Please enter your Twitter API Bearer Token in the sidebar to use this app.")
    
    st.info("""
    ### Financial Twitter Sentiment Analyzer
    
    This app analyzes the sentiment of tweets about financial topics, stocks, and cryptocurrencies.
    
    **Features:**
    - Specialized financial sentiment analysis
    - Bullish/Bearish classification
    - Engagement metrics analysis
    - Sentiment trends over time
    
    Click "Analyze Market Sentiment" to start.
    """)
    