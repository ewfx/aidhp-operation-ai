import random
from stockRecommender import *
from riskScoreCalculator import *

responses = {
    "hello": ["Hi, Please provide your CustomerId"],
    "hi": ["Hi, Please provide your CustomerId"],
    "how are you": ["I'm doing great!", "I'm here to assist you."],
    "bye": ["Goodbye!", "Take care!"],
    "CustomerId": ["Do you have investment plans? (yes/no)"],
    
}

userContext = {}
customerID = ''
investmentAmt = 0
tenure = 0
returnsPer = 0


def getRecommendedStocksAndAssets():
    # customerDataRecord = fetchCustomerData(userContext['customerID'])
    customerDataRecord = {
  "Customer_ID": 100009,
  "Name": "Customer_9",
  "Age": 75,
  "Gender": "Female",
  "Location": "Rural",
  "Annual_Income": 193226,
  "Has_Loan": "Yes",
  "Loan_Amount": 422772,
  "Monthly_Expenses": 7408,
  "Credit_Score": 11181,
  "Savings": 126017,
  "Debt": 586,
  "Employment_Status": "Employed",
  "Risk_Score": 36.68
}

    # risk_score = predict_risk(customerDataRecord)
    customerDataRecord["Risk_Score"] = 36.68
    return recommend_stock(customerDataRecord)
    

def get_response(user_input):       
    user_input = user_input.lower()
    
    for key in responses.keys():
        if key in user_input:
            userContext['question'] =  responses[key]
            return random.choice(responses[key])
    
    #print('userinput:'+user_input)
    previousQuestion = str(userContext['question'])
    #print('previousQuestion:'+ previousQuestion)

    nextQuestion = "Hi" 
    if 'CustomerId' in previousQuestion:
        userContext['customerID']=user_input
        nextQuestion = "Do you have investment plans? (yes/no)"
        
    elif previousQuestion == "Do you have investment plans? (yes/no)":
        nextQuestion = "How much you want to invest?"
        
    elif previousQuestion == "How much you want to invest?":
        nextQuestion = "Tenure?"
        investmentAmt=user_input
        
    elif previousQuestion == "Tenure?":
        nextQuestion = "Expected Returns (in percentage)?"
        tenure = user_input
        
    elif previousQuestion == "Expected Returns (in percentage)?":
        returnsPer=user_input
        return getRecommendedStocksAndAssets()

    print("nextquestion:"+ nextQuestion)
    if nextQuestion == "Hi":
        return "I'm sorry, I didn't understand that."

    else:
        userContext['question']  = nextQuestion
        return nextQuestion
    
    