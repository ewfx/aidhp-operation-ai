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
        f"The stock’s **volatility of {stock['volatility']}** matches well with the investor’s risk tolerance of {investor['Risk_Score']}.",
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
# print(recommend_stocks(investor_profile))
