from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd 
from joblib import load


app = FastAPI()

class Dep(BaseModel):
    duration: float
    campaign: int
    contact_unknown: int
    housing_yes: int
    poutcome_success: int
    balance: float
    contact_cellular: int
    day: int
    month_may: int
    loan_no: int




model = load('model10_classifier_only.joblib')

@app.get("/")
async def root():
    return {"message": "Deposit Prediction API. Use POST /predict to get predictions."}


@app.post("/predict")
async def predict_deposit(data: Dep):
    try:
        df = pd.DataFrame([data.dict()])
        
        proba = model.predict_proba(df)[0, 1]
        pred = model.predict(df)[0]
        return{
                "status": "success",
                "prediction": int(pred),
                "probability": float(proba)
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


