import joblib
import pandas as pd


model = joblib.load("../Risk_analyzer/risk_score_model_v3.pkl")
df = pd.read_csv("../Risk_analyzer/banking_risk_dataset.csv")

class CustomerData():
    Age: int
    Gender: str
    Region: str
    Income: float
    Existing_Loans: str
    Loan_Amount: float
    Credit_Card_Debt: float
    Savings: float
    Investments: float
    Credit_Score: float
    Employment_Status: str

def fetchCustomerData(customerId:int):
    return df.loc[df['Customer_ID'] == 100009]
    



def predict_risk(customerDBRecord:any):
    
    data: CustomerData = {}
    data.Age= customerDBRecord["Age"]
    data.Gender = customerDBRecord["Gender"]
    data.Region = customerDBRecord["Region"]
    data.Income = customerDBRecord["Income"]
    data.Existing_Loans = customerDBRecord["Existing_Loans"]
    data.Loan_Amount = customerDBRecord["Loan_Amount"]
    data.Credit_Card_Debt = customerDBRecord["Credit_Card_Debt"]
    data.Savings = customerDBRecord["Savings"]
    data.Investments = customerDBRecord["Investments"]
    data.Credit_Score = customerDBRecord["Credit_Score"]
    data.Employment_Status = customerDBRecord["Employment_Status"]

    input_data = pd.DataFrame([data.dict()])
    risk_score = model.predict(input_data)[0]
    return {"Risk_Score": round(risk_score, 2)}