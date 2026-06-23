from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize our web app framework
app = FastAPI(title="Medical Insurance Cost Predictor API")

# 2. Load the trained brain (our saved model file)
model = joblib.load("medical_insurance_model.pkl")

# 3. Create a welcoming Home Route (so we avoid 'Not Found' pages)
@app.get("/")
def home():
    return {"message": "Welcome to the Medical Insurance Predictor API! Head over to /docs to test it out."}

# 4. Define the Exact Menu Layout (Pydantic Class)
# This strictly validates the incoming JSON data from our user interface
class MedicalFeatures(BaseModel):
    age: int
    bmi: float
    children: int
    sex_male: bool       # Translates automatically to 1 or 0
    smoker_yes: bool     # Translates automatically to 1 or 0
    region_northwest: bool
    region_southeast: bool
    region_southwest: bool

# 5. Create the Prediction Postman (Endpoint)
@app.post("/predict")
def predict_charges(data: MedicalFeatures):
    # Convert incoming user values directly into a structured NumPy list
    input_data = np.array([[
        data.age,
        data.bmi,
        data.children,
        int(data.sex_male),
        int(data.smoker_yes),
        int(data.region_northwest),
        int(data.region_southeast),
        int(data.region_southwest)
    ]])
    
    # Send it to the model to get the logged prediction
    log_pred = model.predict(input_data)
    
    # Reverse the log transformation using exponential minus 1
    actual_charges = np.expm1(log_pred)[0]
    
    return {"predicted_medical_charges": f"${actual_charges:,.2f}"}
