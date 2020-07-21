from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

app = Flask(__name__)

# def ValuePredictor(to_predict_list): 
#     to_predict = np.array(to_predict_list).reshape(1, 3) 
#     result = loaded_model.predict(to_predict) 
#     return result[0] 

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST':
        
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

        to_predict_list = request.form.to_dict()
         
        to_predict_list = list(to_predict_list.values())
        
        data = np.array(to_predict_list)

        data = data.reshape(1,3)
         
        result = loaded_model.predict(data)
                
        if int(result)== 1: 
            myprediction ='yes, this person tends to buy based on social media ads'
        else: 
            myprediction ='no, this person does not tend to purchase based on social media ads'        
        return render_template("result.html", prediction = myprediction) 


if __name__ == "__main__":
    app.run(debug=True)