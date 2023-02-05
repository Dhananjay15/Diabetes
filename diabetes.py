import numpy as np
import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI
from diabetes_data import Diabetes
from sklearn.preprocessing import StandardScaler


app = FastAPI()

pickle_in = open("svc_model.pkl","rb")
classifier=pickle.load(pickle_in)

sc = StandardScaler()

@app.post('/predict_diabetes')
def predict_diabetes(data):
    data = data.dict()
    Pregnancies = int(data['Pregnancies'])
    Glucose	= int(data['Glucose'])
    BloodPressure = int(data['BloodPressure'])
    SkinThickness = int(data['SkinThickness'])
    Insulin	= int(data['Insulin'])
    BMI	= float(data['BMI'])
    DiabetesPedigreeFunction = float(data['DiabetesPedigreeFunction'])
    Age = int(data['Age'])
    
    data = sc.fit_transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    
    prediction = classifier.predict(data)
    
    if(prediction[0]>0.5):
        prediction="Diabetic"
    else:
        prediction="Not Diabetic"
    return {'prediction': prediction}


        
        
##uvicorn diabetes:app --reload
