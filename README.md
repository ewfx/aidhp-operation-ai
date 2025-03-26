# 🚀 -  GOLDEN BASKET - AI -POWERE D PERSONALIZED RECOMMENDATION SYSTEM

## 📌 Table of Contents

Introduction	
Problem Statement
Our Solution 
What It Does
Introducing “Golden Basket”	
Technical Details: RUG, LLM, Regression Model in Action	
RAG Architecture Overview (TinyLlama + FAISS):	
LLM – Large Language Model:	
Personalized Investment Basket Recommendation Engine	
GenAI-Powered Customer Interaction	
Challenges:	
Conclusion and Future Roadmap	



## Introduction 

In a rapidly evolving financial landscape, customer expectations from their banking partners have transformed significantly.
They now seek not just transactional services, but holistic financial experiences that guide them toward financial well-being and wealth creation.
The vision of "Golden Basket" is to offer a GenAI-powered, highly personalized and customized wealth management platform that enhances the banking experience and enables customers to make informed investment decisions.
 
##Problem Statement
 
Before onboarding onto the investment portal of a Bank or NBFC, every customer undergoes a risk assessment, and investment recommendations are provided based on their risk appetite. 
However, the current recommendation engine applies a standardized approach, offering uniform suggestions to customers with similar risk scores rather than catering to individual preferences.
To address this, there is a need for a hyper-personalized recommendation engine that delivers investment suggestions uniquely tailored to each customer's specific profile and needs.

 
## ⚙️Our Solution
 
 Golden Basket acts as a digital financial companion that understands the customer's financial dreams and risk appetite, offering tailored recommendations in stocks,
 mutual funds, and investment baskets using AI-driven insights. The integration of advanced technologies such as Risk Understanding Graphs (RUG), 
 Large Language Models (LLMs), and regression-based risk profiling ensures precision and personalization.


 
## ⚙️ What It Does
 
 Golden Basket is a GenAI-Powered Personalized Wealth Management system that enhances the banking experience and delivers customized investment solutions.

Key Features:
o	Risk Score Prediction using Regression Modelling
o	Personalized Investment Basket generation
o	Natural Language Explanation using LLM
o	Customer Need Analysis with RUG (Risk Understanding Graph)
o	Investment Goal Mapping
o	AI Storytelling – Explainable portfolio rationale
o	Multi-Channel Delivery – Mobile, Web, Branch Advisor Assistance

## 🛠️ How We Built It

Technical Details: RUG, LLM, Regression Model in Action

RAG Architecture Overview (TinyLlama + FAISS):

RAG stands for “Retrieval-Augmented Generation.” It’s a powerful architecture used to enhance the performance and accuracy of Large Language Models (LLMs) .
RAG = Retrieval + Generation

Stock Data 
To fetch stock ticker, we used - yfinance . It is a Python library that allows you to fetch real-time and historical stock market data from Yahoo Finance.

Stock Analysis 
Stock ranking involves assigning a score to each stock based on multiple financial factors (e.g., price movements, volume, volatility, earnings). LGBMRegressor can be trained to predict future stock returns or an overall performance score, which is then used for ranking. 

Rule Engine 
A Rule Engine in stock recommendation is a logic-based system that applies predefined rules to recommend stocks based on various financial, technical, and sentiment factors. Unlike machine learning models, a rule engine explicitly encodes decision-making criteria and executes them dynamically to filter and rank stocks based on risk score and volatility based on  customer profile.

Rule Engine Output:
Investor-Stock Scoring System  assigns a score to each stock based on how well it matches an investor's profile using rules

	Investor Factors like  Net Worth, Liquidity, Risk Score, Region, Assets and  Mortgage Debt
	Stock Factors like  Volatility, P/E Ratio, Market Cap, Sharpe Ratio, Sector Match

LLM – Large Language Model:

We use LLM Model TinyLlama

This is used to 
•	Convert AI insights into human-readable language
•	Provide portfolio storytelling
•	Answer "Why is this fund for me?" in simple terms

## 🚧 Challenges We Faced

While the solution promises enhanced customer experience and intelligent portfolio recommendations, it also requires overcoming several technical complexities behind the scenes. From integrating large-scale financial datasets to ensuring real-time, context-aware responses via Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), 
each layer of the platform presents unique development, scalability, and compliance hurdles.

•	Choosing between hosted vs self-hosted LLMs (e.g., OpenAI vs Mistral vs LLaMA). LLMs like GPT-4 have token limits (e.g., 8k/32k) — we  can't pass unlimited RAG context into it. We  must optimize prompt + retrieval.
•	We  need to design consistent prompts that include user intent, retrieved context, and formatting instructions (table/chart/text).
•	Choosing Embedding Model - OpenAI, Cohere, HuggingFace models vary in quality, language support, and cost.
•	FAISS works for small scale, we faced challenges when using for large scale data
•	Retrieval + LLM generation + formatting must happen in milliseconds — this is tricky when we use  personnel  Laptops for development.
•	Model Fusion - Combining regression risk scoring with RAG retrieval and LLM generation was bit challenging
•	Optimizing compute resources for regression scoring + retrieval + generation pipelines was a big challenge

 ## 🏃 How to Run
 
1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/aidhp-operation-ai.git
   ```
2. Install dependencies  
   !pip install yfinance transformers torch vaderSentiment pandas numpy scikit-learn xgboost lightgbm
   !pip install sentencepiece 
   !pip install --upgrade pandas
   !pip install --upgrade lightgbm
   !pip install --upgrade dask


   ```
3. Run the project  
   ```sh
   python golden_basket.py
   
 ## 🏗️ Tech Stack
   
- 🔹 Frontend: HTML, CSS, React.js
- 🔹 Backend: Flask, Python script
- 🔹 Database: Open end APIs - Yfinance API to get data
- 🔹 Other: Ilama LLM Model / LGBM Model / RandomForest Regressor Model

## 👥 Team
- Vijayalakshmi Gunasekaran
- Chaitanya Kumar ,Srikakolapu
- Mangaiyarkarasi S
- Balaji Gnanasekaran
