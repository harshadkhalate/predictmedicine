import pandas as pd
from pandas import read_csv
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load your pre-trained model
gnb = pickle.load(open('drugmodel.pkl', 'rb'))

# Create a FastAPI app
app = FastAPI()

# Define the input schema using Pydantic
class DrugRequest(BaseModel):
    Disease: str
    Gender: str
    Age: int

# Define the endpoint for prediction
@app.post("/predictmedicine")
def predict_drug(data: DrugRequest):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([[data.Disease, data.Gender, data.Age]], columns=['Disease', 'Gender', 'Age'])
    
    # Map string values to numerical values as done in your preprocessing
    input_data.replace({'Disease': {'Acne': 0, 'Allergy': 1, 'Diabetes': 2, 'Fungal infection': 3,
                                    'Urinary tract infection': 4, 'Malaria': 5, 'Migraine': 6, 'Hepatitis B': 7,
                                    'AIDS': 8},
                        'Gender': {'Female': 0, 'Male': 1}}, inplace=True)
    
    # Make predictions
    prediction = gnb.predict(input_data)

    # Return the prediction
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
