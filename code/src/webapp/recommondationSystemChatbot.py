import random
from riskScoreCalculator import *
from recommender_service import *

responses = {
    "hello": ["Hi, Please provide your CustomerId"],
    "hi": ["Hi, Are you Existing Customer? (yes/no)"],    
    "how are you": ["I'm doing great!", "I'm here to assist you."],
    "bye": ["Goodbye!", "Take care!"],
    "Ready": ["Do you have investment plans? (yes/no)"],
    
}

userContext = {}
customerID = ''
investmentAmt = 0
tenure = 0
returnsPer = 0


def getRecommendedStocksAndAssets():
    customerDataRecord = fetchCustomerData(int(userContext['customerID']))
    return recommend_stocks(customerDataRecord)
    # # risk_score = predict_risk(customerDataRecord)
    # # customerDataRecord["Risk_Score"] = risk_score

    # # print(customerDataRecord)

    # return "Sample records"


    

def get_response(user_input):       
    user_input = user_input.lower()

    if user_input == 'repeat':
        return getRecommendedStocksAndAssets()
    
    for key in responses.keys():
        if key in user_input:
            userContext['question'] =  responses[key][0]
            return random.choice(responses[key])
    
    #print('userinput:'+user_input)
    previousQuestion = str(userContext['question'])
    #print('previousQuestion:'+ previousQuestion)

    nextQuestion = "Hi" 
    if previousQuestion == "Hi, Are you Existing Customer? (yes/no)":
        if user_input == "yes":
            nextQuestion = "Please provide your CustomerId"
        else:
            return "Please share your details in the form. After submitting please say 'Ready'" \
            "http://127.0.0.1:5000/newCustomer"


    elif previousQuestion == "Please provide your CustomerId":
        userContext['customerID']=user_input
        nextQuestion = "Do you have investment plans? (yes/no)"
        
    elif previousQuestion == "Do you have investment plans? (yes/no)":
        nextQuestion = "How much you want to invest?"
        
    elif previousQuestion == "How much you want to invest?":
        nextQuestion = "Tenure?"
        userContext['investmentAmt']=user_input
        
    elif previousQuestion == "Tenure?":
        nextQuestion = "Expected Returns (in percentage)?"
        userContext['tenure'] = user_input
        
    elif previousQuestion == "Expected Returns (in percentage)?":
        userContext['returnsPer']=user_input
        return getRecommendedStocksAndAssets()
    
   

    print("nextquestion:"+ nextQuestion)
    if nextQuestion == "Hi":
        return "I'm sorry, I didn't understand that."

    else:
        userContext['question']  = nextQuestion
        return nextQuestion
    
    