from flask import Flask, request, render_template, redirect
from recommondationSystemChatbot import *


app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template('recommendation-chatbot.html')

# @app.route('/predict')
# def fetchRecommendations():
    
#     print(request.args.get("customerID"))
#     print(request.args.get("invAmt"))
#     print(request.args.get("tenure"))
#     print(request.args.get("returnsPersent"))
#     customerID = request.args.get("customerID")
#     invAmt = request.args.get("invAmt")
#     tenure = request.args.get("tenure")
#     returnsPersent = request.args.get("returnsPersent")
    
#     riskScore = getCustomerPortFolioScore(customerID)
#     print(riskScore)

#     recommendations = getRecommendedStocksAndAssets(riskScore,customerID,invAmt, tenure, returnsPersent)
    
#     return recommendations



@app.route('/chatbot')
def chatBotInteration():
    print(request.args.get("userMessage"))
    return get_response(request.args.get("userMessage"))

if __name__ == '__main__':
     app.run()