#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance transformers torch vaderSentiment pandas numpy scikit-learn xgboost lightgbm')

get_ipython().system('pip install sentencepiece ')
get_ipython().system('pip install --upgrade pandas')
get_ipython().system('pip install --upgrade lightgbm')
get_ipython().system('pip install --upgrade dask')


# In[3]:


import torch
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



# In[99]:



# Load TinyLlama model and tokenizer once (to avoid reloading)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tinyllama_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")


# # Save TinyLama Pickle file

# In[100]:


import requests
from bs4 import BeautifulSoup
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# **Fetch Stock Info**

# In[5]:


# Load csv File
df = pd.read_csv("../recommender_model/dataset/stock_data.csv")

# Rename columns using a dictionary
df.rename(columns={"Ticker Symbol": "Ticker", "Volatility": "Volatility",
                  "Net Cash Flow" : "Market Cap", "GICS Sector": "Sector",
                  "P/E Ratio": "P/E Ratio", "Sharpe_Ratio": "Sharpe Ratio",
                  "Dividend_Yield (%)": "Dividend_Yield"}, inplace=True)

# Select 5 Random Rows
random_stocks = df.sample(n=4, random_state=42)  # Set random_state for reproducibility
# random_stocks


random_tickers = random_stocks['Ticker'].tolist()


# In[6]:



# Define Stock Universe
stock_list = random_tickers

# Fetch Stock Data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "sector": info.get("sector", "Unknown"),
        "market cap": info.get("marketCap", 0),
        "volatility": info.get("beta", 1),
        "p/e ratio": info.get("trailingPE", None),
        "stock_price": info.get("currentPrice", None),
        "sharpeRatio": info.get("sharpeRatio", 0.5),
        "dividend_yield":info.get("dividendYield", 0)
    }


def get_stock_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news  # Fetch news data

#     print("News Response:", news)  # Debugging print

    if not news:  # Check if news is empty
        return ["No news available for this stock"]

    headlines = [article.get("title", "No title found") for article in news[:2]]
    return headlines


# In[103]:


# print(get_stock_news("AAPL"))


# In[104]:



# Function to analyze sentiment using Llama LLM
def analyze_sentiment_llm(headlines):
    responses = []
    
    for headline in headlines:
        prompt = f"Analyze the sentiment (Positive, Negative, Neutral) of this news: '{headline}'."
        
        inputs = tokenizer(prompt, return_tensors="pt")
        output = tinyllama_model.generate(**inputs, max_length=30)
        
#         print("Raw Model Output:", output)  # Debugging step
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        responses.append(response_text.strip())

    return responses


# **Stock Sentiment Analysis**

# In[105]:





# Function to analyze stock sentiment
def analyze_stock_sentiment(ticker):
    # Fetch stock-related news headlines
    headlines = get_stock_news(ticker) or []  # Ensure it's not None

    if not headlines:  # If no headlines are available, return default sentiment scores
        return {"Stock": ticker, "POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    # Analyze sentiment using LLM
    llm_sentiments = analyze_sentiment_llm(headlines) or []  # Ensure it's a list

    # Initialize sentiment score counters
    sentiment_scores = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    # Count occurrences of each sentiment
    for sentiment in llm_sentiments:
        sentiment = sentiment.lower()  # Convert to lowercase for consistency

        if "positive" in sentiment:
            sentiment_scores["POSITIVE"] += 1
        elif "negative" in sentiment:
            sentiment_scores["NEGATIVE"] += 1
        else:
            sentiment_scores["NEUTRAL"] += 1

    return {"Stock": ticker, **sentiment_scores}



# In[106]:


# analyze_stock_sentiment("SWN")


# In[107]:


# headlines = get_stock_news("SWN") or []
# print("Fetched headlines:", headlines)  # Debugging print


# **Collect stock data and prepare dataset**

# In[108]:




# Function to collect stock data and prepare dataset
def collect_stock_data(stock_list):
    stock_data = []

    for ticker in stock_list:
        sentiment = analyze_stock_sentiment(ticker)
        fundamentals = get_stock_data(ticker)

        data = {**sentiment, **fundamentals}
        stock_data.append(data)

    df = pd.DataFrame(stock_data)
    df["Sentiment Score"] = df["POSITIVE"] - df["NEGATIVE"]
    df.dropna(inplace=True)  # Remove NaN values

    return df


# In[109]:


# collect_stock_data(['SWN', 'ETN', 'ES', 'BBT', 'FBHS', 'NTRS', 'KO', 'SPGI', 'ZBH', 'AMT'])


# **LGBM Model**

# In[110]:



# Train LightGBM for Stock Ranking
def train_stock_ranking_model(df):
    X = df[["Sentiment Score", "p/e ratio", "market cap", "volatility"]]
    y = df["Sentiment Score"]  # Target: Higher sentiment = better rank

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    df["Recommendation Score"] = model.predict(X)

    return df.sort_values(by="Recommendation Score", ascending=False)


# **Rule Engine**

# In[111]:


import pandas as pd

def calculate_investor_stock_score_old(investor, stock_df):
    """
    Assigns a score (out of 100) to a stock based on investor profile.
    
    Parameters:
    - investor (dict): Investor profile
    - stock (dict): Stock details
    
    Returns:
    - int: Final score for the stock
    """
    score = 0.0
    risk_score = 0.0
    liquidity = 0.0
    volatility = 0.0
    market_cap = 0.0
    sharpeRatio = 0.0
    p_e_ratio =0.0
    dividend_yield=0.0
    
#     print(stock)
    
    # Risk Score Match (25%) - Scaled from 1 to 100
    if isinstance(investor["risk_score"], (int, float)): 
        risk_score = investor["risk_score"]
    if isinstance(investor["liquidity"], (int, float)): 
        liquidity = investor["liquidity"]          
    if isinstance(stock["volatility"], (int, float)): 
        volatility = stock["volatility"]
    if isinstance(stock["market cap"], (int, float)): 
        market_cap = stock["market cap"]
    if isinstance(stock["p/e ratio"], (int, float)): 
        p_e_ratio = stock["p/e ratio"]
    if isinstance(stock["sharpeRatio"], (int, float)): 
        sharpeRatio = stock["sharpeRatio"]
        
    if isinstance(stock["dividend_yield"], (int, float)): 
        dividend_yield = stock["dividend_yield"]
    
    
    if risk_score <= 20 and volatility < 0.3:
        score += 25
    elif 21 <= risk_score <= 40 and volatility < 0.6:
        score += 25
    elif 41 <= risk_score <= 60 and 0.6 <= volatility <= 1.2:
        score += 25
    elif 61 <= risk_score <= 80 and volatility <= 2.0:
        score += 25
    elif 81 <= risk_score <= 100:
        score += 25  # High-risk investors can take any stock

    # Liquidity Match (20%)
    if liquidity > market_cap * 0.0001:
        score += 20

    if stock["sector"].empty and investor["preferred_sectors"].any():
        # Sector Preference (15%)
        if investor["preferred_sectors"] == stock["sector"]:
            score += 15

    # Market Cap Stability (15%)
    if market_cap > 50_000_000_000:  # Large-cap stock
        score += 15
    elif market_cap > 10_000_000_000:  # Mid-cap stock
        score += 10

    # P/E Ratio (10%) - Favor stocks with a reasonable P/E ratio (10-30)
    if 10 <= p_e_ratio <= 30:
        score += 10

    # Sharpe Ratio (10%) - Higher Sharpe Ratio gets higher points
    if sharpeRatio > 1.0:
        score += 10
    elif sharpeRatio > 0.5:
        score += 5

   # Dividend Yield Stability (5%) - Consistent dividend payers score higher
    if dividend_yield >= 2.0:
        score += 5 
    return score

# Example Investor Profile
investor = {
    "net_worth": 250000,
    "liquidity": 50000,
    "risk_score": 3,
    "region": "US",
    "assets": 500000,
    "mortgage": 100000,
    "preferred_sector": "Healthcare"

}

# Example Stock Data
# stocks = [
#     {"ticker": "AAPL", "sector": "Technology", "market_cap": 2900000000000, "volatility": 1.2, "p_e_ratio": 28, "sharpe_ratio": 1.1, "dividend_yield": 8.51},
#     {"ticker": "JNJ", "sector": "Healthcare", "market_cap": 430000000000, "volatility": 0.7, "p_e_ratio": 18, "sharpe_ratio": 0.9, "dividend_yield": 9.83},
#     {"ticker": "TSLA", "sector": "Automotive", "market_cap": 850000000000, "volatility": 2.0, "p_e_ratio": 60, "sharpe_ratio": 0.8, "dividend_yield": 5.6},
#     {"ticker": "KO", "sector": "Consumer Defensive", "market_cap": 295000000000, "volatility": 0.5, "p_e_ratio": 27, "sharpe_ratio": 0.5, "dividend_yield": 5.51},
# ]

# Calculate scores for all stocks
# stock_scores = [{"ticker": stock["ticker"], "score": calculate_investor_stock_score(investor, stock)} for stock in stocks]

# # Convert to DataFrame and sort by best score
# df_scores = pd.DataFrame(stock_scores).sort_values(by="score", ascending=False)


# In[112]:


import pandas as pd

def calculate_investor_stock_score(investor, stock):
    """Assigns a score (out of 100) to a stock based on investor profile."""
    risk_score = pd.to_numeric(investor.get("risk_score", 0), errors="coerce")
    liquidity = pd.to_numeric(investor.get("liquidity", 0), errors="coerce")
    volatility = pd.to_numeric(stock.get("volatility", 0), errors="coerce")
    market_cap = pd.to_numeric(stock.get("market_cap", 0), errors="coerce")
    p_e_ratio = pd.to_numeric(stock.get("p_e_ratio", 0), errors="coerce")
    sharpe_ratio = pd.to_numeric(stock.get("sharpeRatio", 0), errors="coerce")
    dividend_yield = pd.to_numeric(stock.get("dividend_yield", 0), errors="coerce")

    sector = stock.get("sector", "")
    preferred_sectors = investor.get("preferred_sectors", [])

    score = 0.0

    # Risk Score Match (25%)
    if risk_score <= 20 and volatility < 0.3:
        score += 25
    elif 21 <= risk_score <= 40 and volatility < 0.6:
        score += 25
    elif 41 <= risk_score <= 60 and 0.6 <= volatility <= 1.2:
        score += 25
    elif 61 <= risk_score <= 80 and volatility <= 2.0:
        score += 25
    elif 81 <= risk_score <= 100:
        score += 25

    # Liquidity Match (20%)
    if liquidity > market_cap * 0.0001:
        score += 20

    # Sector Preference (15%)
    if sector and isinstance(preferred_sectors, list) and sector in preferred_sectors:
        score += 15

    # Market Cap Stability (15%)
    if market_cap > 50_000_000_000:
        score += 15
    elif market_cap > 10_000_000_000:
        score += 10

    # P/E Ratio (10%)
    if 10 <= p_e_ratio <= 30:
        score += 10

    # Sharpe Ratio (10%)
    if sharpe_ratio > 1.0:
        score += 10
    elif sharpe_ratio > 0.5:
        score += 5

    # Dividend Yield (5%)
    if dividend_yield >= 2.0:
        score += 5

    return round(score, 2)

# Example Investor Profile
# investor = {
#     "risk_score": 68.5,
#     "liquidity": 50000,
#     "preferred_sectors": ["Healthcare", "Technology"]
# }

# # Example Stock DataFrame
# stock_df = pd.DataFrame([
#     {"sector": "Healthcare", "volatility": 0.8, "market_cap": 100_000_000_000, "p_e_ratio": 25, "sharpeRatio": 1.2, "dividend_yield": 2.5},
#     {"sector": "Technology", "volatility": 1.1, "market_cap": 200_000_000_000, "p_e_ratio": 30, "sharpeRatio": 0.9, "dividend_yield": 1.8},
#     {"sector": "Energy", "volatility": 1.5, "market_cap": 50_000_000_000, "p_e_ratio": 15, "sharpeRatio": 1.3, "dividend_yield": 3.0},
# ])

# Apply function to each row
# stock_df["score"] = stock_df.apply(lambda stock: calculate_investor_stock_score(investor, stock), axis=1)

# print(stock_df[["sector", "score"]])


# **Stock Recommneder**

# In[113]:




# Recommend Stocks Based on Investor Profile
def recommend_stocks(net_worth, liquidity, risk_score, region, assets, mortgage, preferred_sectors):
    investor_profile= {
          "net_worth": net_worth,
          "liquidity": liquidity,
          "risk_score": risk_score,
          "mortgage": mortgage,
          "region": region,
          "assets": assets,
          "preferred_sectors": preferred_sectors
      }

    stock_data = collect_stock_data(stock_list)

    ranked_df = train_stock_ranking_model(stock_data)

    stock_data_df = pd.DataFrame(ranked_df).dropna()
 
    # Apply function to each row
    stock_data_df["investor_stock_score"] = stock_data_df.apply(lambda stock: calculate_investor_stock_score(investor, stock), axis=1)
    top_5_stocks = stock_data_df.sort_values(by="investor_stock_score", ascending=False).head(5)
    stocks_df = pd.DataFrame(top_5_stocks).dropna()

    return stocks_df


# In[114]:


# recommend_stocks(250000,50000, 67.5,"US",500000,100000,"Healthcare")


# **AI Explanation**

# In[115]:


def generate_stock_explanation_prompt(stock_info, investor_profile):
    """
    Creates a prompt to generate a friendly, engaging, and concise three-line stock recommendation.
    """
    

#     prompt1 = f"""
#     Investor Profile:
#     - Net Worth: {investor_profile['net_worth']} USD
#     - Liquidity: {investor_profile['liquidity']} USD
#     - Risk Score: {investor_profile['risk_score']}
#     - Mortgage Debt: {investor_profile['mortgage']} USD
#     - Region: {investor_profile['region']}
#     - Preferred Sector: {investor_profile['preferred_sectors']}

#     Given this investor profile, explain why {stock_info['ticker']} is a suitable stock recommendation.
#     Stock Data:
#     - Sector: {stock_info['sector']}
#     - Market Cap: {stock_info['market cap']}
#     - Volatility: {stock_info['volatility']}
#     - P/E Ratio: {stock_info['p/e ratio']}
#     - Sharpe Ratio: {stock_info['sharpeRatio']}
#     - Dividend Yield: {stock_info['dividend_yield']}
#     - Sentiment Score: {stock_info['Sentiment Score']}
#     - Recommendation Score: {stock_info['Recommendation Score']}

#     Provide an easy-to-understand explanation.
#     """
    
    prompt = f"""
    You are an expert financial advisor providing stock recommendations in a friendly and engaging way.

    Investor Profile:
    - Net Worth: {investor_profile['net_worth']} USD
    - Liquidity: {investor_profile['liquidity']} USD
    - Risk Score: {investor_profile['risk_score']}
    - Mortgage Debt: {investor_profile['mortgage']} USD
    - Region: {investor_profile['region']}
    - Preferred Sector: {investor_profile['preferred_sectors']}

    Stock Recommendation: {stock_info['ticker']}
    - Sector: {stock_info['sector']}
    - Market Cap: {stock_info['market cap']}
    - Volatility: {stock_info['volatility']}
    - P/E Ratio: {stock_info['p/e ratio']}
    - Sharpe Ratio: {stock_info['sharpeRatio']}
    - Dividend Yield: {stock_info['dividend_yield']}
    - Sentiment Score: {stock_info['Sentiment Score']}
    - Recommendation Score: {stock_info['Recommendation Score']}

    **Your task:** Explain in three lines step by step why this stock is a good fit for the investor in a friendly and informative way. Keep the response engaging, avoiding excessive jargon.
    """
    
#     prompt = f"""
# You are an expert financial advisor providing stock recommendations in a **friendly and engaging way**.

# ### **Investor Profile**
# - **Net Worth**: {investor_profile['net_worth']} USD
# - **Liquidity**: {investor_profile['liquidity']} USD
# - **Risk Score**: {investor_profile['risk_score']}
# - **Mortgage Debt**: {investor_profile['mortgage']} USD
# - **Region**: {investor_profile['region']}
# - **Preferred Sector**: {investor_profile['preferred_sectors']}

# ### **Stock Recommendation: {stock_info['ticker']}**
# - **Sector**: {stock_info['sector']}
# - **Market Cap**: {stock_info['market cap']}
# - **Volatility**: {stock_info['volatility']}
# - **P/E Ratio**: {stock_info['p/e ratio']}
# - **Sharpe Ratio**: {stock_info['sharpeRatio']}
# - **Dividend Yield**: {stock_info['dividend_yield']}
# - **Sentiment Score**: {stock_info['Sentiment Score']}
# - **Recommendation Score**: {stock_info['Recommendation Score']}

# ### **Why this stock?**
# Provide exactly **three engaging lines** explaining why this stock is a great fit for the investor.  
# Each line should highlight a **key reason** (financial strength, risk match, growth potential, etc.)  
# Make it friendly, informative, and free of excessive jargon.  

# **Example Output:**
# 1. This stock is in the investor's preferred sector, ensuring alignment with their interests.  
# 2. Its **high Sharpe Ratio** means strong returns for the risk taken, ideal for their risk score.  
# 3. With a **solid dividend yield**, this stock provides a steady income stream, adding stability.  
# """
    return prompt.strip()


# In[116]:



# AI Justification for Stock Selection
def ai_stock_explanation(stock_info, investor_profile):

    
    prompt = generate_stock_explanation_prompt(stock_info, investor_profile)
    print("prompt", prompt) 
    inputs = tokenizer(prompt, return_tensors="pt")
    response = tinyllama_model.generate(**inputs, max_length=2000)

    recommendation = tokenizer.decode(response[0], skip_special_tokens=True)
    # Remove Prompt from Output
    clean_output = recommendation.replace(prompt, "").strip()

#     print(clean_output)
    return clean_output


# In[117]:


investor_profile= {
        "net_worth": 250000,
        "liquidity": 50000,
        "risk_score": 3,
        "mortgage": 100000,
        "region": "US",
        "assets": 150000,
        "preferred_sectors": "Healthcare"
}

best_stocks = recommend_stocks(investor_profile["net_worth"], investor_profile["liquidity"],
                               investor_profile["risk_score"], investor_profile["region"], investor_profile["assets"],
                               investor_profile["mortgage"], investor_profile["preferred_sectors"])
# print("best_stocks")
# best_stocks
# stock_data = get_stock_data("AAPL")
# stock_data
stock_explanation = ai_stock_explanation(best_stocks, investor_profile)
# why_this_stock_section = stock_explanation.split("### **Why this stock?**")[-1].strip()
# print(why_this_stock_section)
stock_explanation


# In[7]:


import random

# Sample stock list with financial data
stocks = [
    {
        "ticker": "AAPL",
        "sector": "Technology",
        "market cap": 2800000000000,
        "volatility": 0.35,
        "p/e ratio": 28.3,
        "sharpeRatio": 1.5,
        "dividend_yield": 0.6,
        "Sentiment Score": 0.82,
        "Recommendation Score": 90
    },
    {
        "ticker": "JNJ",
        "sector": "Healthcare",
        "market cap": 428000000000,
        "volatility": 0.45,
        "p/e ratio": 18.5,
        "sharpeRatio": 1.2,
        "dividend_yield": 2.8,
        "Sentiment Score": 0.75,
        "Recommendation Score": 85
    },
    {
        "ticker": "TSLA",
        "sector": "Automotive",
        "market cap": 700000000000,
        "volatility": 1.1,
        "p/e ratio": 73.2,
        "sharpeRatio": 0.9,
        "dividend_yield": 0.0,
        "Sentiment Score": 0.88,
        "Recommendation Score": 78
    },
    {
        "ticker": "MSFT",
        "sector": "Technology",
        "market cap": 2600000000000,
        "volatility": 0.3,
        "p/e ratio": 32.1,
        "sharpeRatio": 1.7,
        "dividend_yield": 1.0,
        "Sentiment Score": 0.9,
        "Recommendation Score": 92
    },
    {
        "ticker": "AMZN",
        "sector": "E-Commerce",
        "market cap": 1700000000000,
        "volatility": 0.6,
        "p/e ratio": 60.2,
        "sharpeRatio": 1.0,
        "dividend_yield": 0.0,
        "Sentiment Score": 0.78,
        "Recommendation Score": 80
    }
]

# Example investor profile
investor_profile = {
    "net_worth": 500000,
    "liquidity": 100000,
    "risk_score": 55,
    "mortgage": 150000,
    "region": "US",
    "preferred_sectors": ["Technology", "Healthcare"]
}

# Function to generate recommendation reasons
def get_recommendation_reasons(stock, investor):
    return [
        f"{stock['ticker']} is in the **{stock['sector']}** sector, aligning with the investor’s preferred industries.",
        f"The stock’s **volatility of {stock['volatility']}** matches well with the investor’s risk tolerance of {investor['risk_score']}.",
        f"With a **market cap of ${stock['market cap']:,}**, this stock is financially stable and suitable for long-term investing.",
        f"A **P/E ratio of {stock['p/e ratio']}** suggests this stock is {'fairly valued' if 10 <= stock['p/e ratio'] <= 30 else 'potentially overvalued'}.",
        f"The **Sharpe Ratio of {stock['sharpeRatio']}** indicates {'strong risk-adjusted returns' if stock['sharpeRatio'] > 1 else 'moderate returns'}.",
        f"The dividend yield of **{stock['dividend_yield']}%** makes this stock attractive for income-focused investors.",
        f"With a **Sentiment Score of {stock['Sentiment Score']}**, market perception is {'positive' if stock['Sentiment Score'] > 0.6 else 'neutral/negative'}.",
        f"A Recommendation Score of **{stock['Recommendation Score']}/100** suggests strong alignment with the investor’s financial profile.",
        f"This stock has a **strong track record** of stable earnings growth, which supports the investor’s wealth-building goals.",
        f"As a company with **low debt and strong cash flow**, {stock['ticker']} offers a financially sound investment opportunity.",
        f"The stock has performed well historically during market downturns, making it a **defensive investment choice** for this investor.",
    ]

# Function to generate recommendations for selected stocks
def recommend_random_stocks_with_reasons(investor, stock):
    reasons = get_recommendation_reasons(stock, investor)
    random_reasons = random.sample(reasons, 3)  # Pick 3 random reasons

    recommendation = f"""### **Why {stock['ticker']}?**
- {random_reasons[0]}
- {random_reasons[1]}
- {random_reasons[2]}"""

    return recommendation

# Function to select random stocks and generate recommendations
def recommend_stocks(investor_profile):
    num_stocks=3 
    selected_stocks = random.sample(stocks, num_stocks)  # Pick random stocks
    recommendations = [recommend_random_stocks_with_reasons(investor_profile, stock) for stock in selected_stocks]
    return "\n\n".join(recommendations)

# Get recommendations for 3 random stocks
print(recommend_stocks(investor_profile))


# In[ ]:




