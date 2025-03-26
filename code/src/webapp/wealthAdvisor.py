# %%
# !pip install yfinance transformers torch vaderSentiment pandas numpy scikit-learn xgboost lightgbm
# !pip install sentencepiece


# %%
import torch
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load LLaMA Model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Adjust based on availability
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")



# %%
import requests
from bs4 import BeautifulSoup
import lightgbm as lgb
from sklearn.model_selection import train_test_split




# %% [markdown]
# **Fetch Stock Info**

# %%
# Load csv File
df = pd.read_csv("stock_data.csv")
# Select 10 Random Rows
random_stocks = df.sample(n=10, random_state=42)  # Set random_state for reproducibility
random_stocks
random_tickers = random_stocks['Ticker Symbol'].tolist()


# %%

# Define Stock Universe
stock_list = random_tickers

# Fetch Stock Data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Ticker": ticker,
        "Sector": info.get("sector", "Unknown"),
        "Market Cap": info.get("marketCap", 0),
        "Volatility": info.get("beta", 1),
        "P/E Ratio": info.get("trailingPE", None),
        "Stock Price": info.get("currentPrice", None),
        "Sharpe Ratio": info.get("sharpeRatio", 0.5)
    }

# Function to fetch recent news headlines for a stock
def get_stock_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for item in soup.find_all("h3", class_="Mb(5px)"):
        headline = item.get_text()
        headlines.append(headline)

    return headlines[:5]  # Limit to top 5 headlines



# %% [markdown]
# **Stock Sentiment Analysis**

# %%


# Function to analyze sentiment using Llama LLM
def analyze_sentiment_llm(headlines):
    responses = []
    for headline in headlines:
        prompt = f"Analyze the sentiment (Positive, Negative, Neutral) of this news: '{headline}'."
        response = model(prompt)
        responses.append(response["choices"][0]["text"].strip())

    return responses

# Function to analyze stock sentiment
def analyze_stock_sentiment(ticker):
    headlines = get_stock_news(ticker)
    if not headlines:
        return {"Stock": ticker, "POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    llm_sentiments = analyze_sentiment_llm(headlines)
    sentiment_scores = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for sentiment in llm_sentiments:
        if "positive" in sentiment.lower():
            sentiment_scores["POSITIVE"] += 1
        elif "negative" in sentiment.lower():
            sentiment_scores["NEGATIVE"] += 1
        else:
            sentiment_scores["NEUTRAL"] += 1

    return {"Stock": ticker, **sentiment_scores}



# %% [markdown]
# **Collect stock data and prepare dataset**

# %%


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


# %% [markdown]
# **LGM Model**

# %%

# Train LightGBM for Stock Ranking
def train_stock_ranking_model(df):
    X = df[["Sentiment Score", "P/E Ratio", "Market Cap", "Volatility"]]
    y = df["Sentiment Score"]  # Target: Higher sentiment = better rank

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    df["Recommendation Score"] = model.predict(X)

    return df.sort_values(by="Recommendation Score", ascending=False)



# %% [markdown]
# **Rule Engine**

# %%
# Match investor profile with stock selection
def match_investor_profile(df, investor_profile):
    df["risk_match"] = 1 - abs(investor_profile["risk_score"] / 10 - df["Volatility"])
    df["sector_match"] = df["Sector"].apply(lambda x: 1 if x in investor_profile["preferred_sectors"] else 0)
    df["Final Score"] = (
        df["Sentiment Score"] * 0.4 + df["risk_match"] * 0.3 + df["sector_match"] * 0.3
    )

    # Profile-Based Filtering
    if investor_profile["risk_score"] >= 80:  # High Risk
        df = df[df["Volatility"] >= 1.0]
    elif investor_profile["risk_score"] <= 30:  # Low Risk
        df = df[df["Volatility"] <= 0.8]

    # if net_worth > 1_000_000:
    #     df = df.sort_values(by="Market Cap", ascending=False)
    # elif net_worth < 100_000:
    #     df = df.sort_values(by="Stock Price", ascending=True)

    # if mortgage > 0 and liquidity < 50_000:
    #     df = df[df["Sharpe Ratio"] >= 1.0]
    return df.sort_values(by="Final Score", ascending=False)


# %% [markdown]
# **Stock Recommneder**

# %%

def recommend_stocks_by_customer(customerData:any):
    return recommend_stocks(customerData["Net_Worth"], customerData["Liquidity"],
                            customerData["Risk_Score"], customerData["Region"],
                            customerData["Assets"], customerData["Mortgage"],
                            customerData["Preferred_Sectors"])

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
#   stock_data = [get_stock_data(stock) for stock in stock_list]
    stock_data = collect_stock_data(stock_list)
    print(stock_data)
    ranked_df = train_stock_ranking_model(stock_data)

    df = pd.DataFrame(stock_data).dropna()

    matched_df = match_investor_profile(df, investor_profile)
    top_5_stocks = matched_df.head(5)
    stocks_df = pd.DataFrame(top_5_stocks).dropna()


    return stocks_df


# %% [markdown]
# **AI Explanation**

# %%

# AI Justification for Stock Selection
def ai_stock_explanation(ticker, stock_info, investor_profile):


    prompt = f"""
    Investor Profile:
    - Net Worth: {investor_profile['net_worth']} USD
    - Liquidity: {investor_profile['liquidity']} USD
    - Risk Score: {investor_profile['risk_score']}
    - Mortgage Debt: {investor_profile['mortgage']} USD
    - Region: {investor_profile['region']}
    - Preferred Sector: {investor_profile['preferred_sectors']}

    Given this investor profile, explain why {ticker} is a suitable stock recommendation.
    Stock Data:
    - Sector: {stock_info['Sector']}
    - Market Cap: {stock_info['Market Cap']}
    - Volatility: {stock_info['Volatility']}
    - P/E Ratio: {stock_info['P/E Ratio']}
    - Sharpe Ratio: {stock_info['Sharpe Ratio']}

    Provide an easy-to-understand explanation.
    """


    inputs = tokenizer(prompt, return_tensors="pt")
    response = model.generate(**inputs, max_length=300)

    recommendation = tokenizer.decode(response[0], skip_special_tokens=True)
    print(recommendation)
#     response = ai_wealth_advisor_model(prompt, max_length=250, num_return_sequences=1)
#     return response[0]["generated_text"]
    return recommendation

# %%
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

stockname = best_stocks['Stock'].iloc[0]
stock_explanation = ai_stock_explanation(stockname,get_stock_data(stockname), investor_profile)
stock_explanation


