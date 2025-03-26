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
    customerRecord = df.loc[df['Customer_ID'] == customerId]
    print(customerRecord)
    return customerRecord
    

def addNewCustomer(customerRecord:any):
    global df
    print(customerRecord)
    new_row = pd.DataFrame(customerRecord, index=df.index+1)
    df = pd.concat([df, new_row], ignore_index=True)

def predict_risk(customerDBRecord:any):
    
    data: CustomerData = {}
    data['Age']= customerDBRecord["Age"]
    data['Gender'] = customerDBRecord["Gender"]
    data['Region'] = customerDBRecord["Region"]
    data['Income'] = customerDBRecord["Income"]
    data['Existing_Loans'] = customerDBRecord["Existing_Loans"]
    data['Loan_Amount'] = customerDBRecord["Loan_Amount"]
    data['Credit_Card_Debt'] = customerDBRecord["Credit_Card_Debt"]
    data['Savings'] = customerDBRecord["Savings"]
    data['Investments'] = customerDBRecord["Investments"]
    data['Credit_Score'] = customerDBRecord["Credit_Score"]
    data['Employment_Status'] = customerDBRecord["Employment_Status"]

    
    #input_data = pd.DataFrame(data, 0)
    risk_score = model.predict(customerDBRecord)[0]

    #input_data = pd.DataFrame([data])
    #risk_score = model.predict(pd.DataFrame(data, 0))[0]
    return {"Risk_Score": round(risk_score, 2)}