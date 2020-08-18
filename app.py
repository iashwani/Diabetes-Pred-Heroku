import numpy as np
import pandas as pd
from flask_cors import cross_origin
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            filename = 'modelForPrediction.sav'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            scalar = pickle.load(open('sandardScalar.sav', 'rb'))
            # predictions using the loaded model file
            prediction = loaded_model.predict(scalar.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))

            print('prediction is', prediction)
            # showing the prediction results in a UI
            if prediction == 1:
                return render_template('positive.html')
            else:
                return render_template('negative.html')

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    #to run locally
    #app.run(host='127.0.0.1', port=8000, debug=True)

    #to run on cloud
	app.run(debug=False) # running the app