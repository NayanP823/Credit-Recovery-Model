from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("risk_model.joblib")

app = FastAPI(title="Risk Scoring API")

class CustomerData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.post("/assess_risk")
def assess_risk(data: CustomerData):
    # Convert input to DataFrame (to match model training input names)
    input_data = pd.DataFrame([data.model_dump()])
    
    # Predict probability of default (class 1)
    try:
        prob = model.predict_proba(input_data)[0][1]
        risk_score = round(prob * 100, 2)
        
        # Determine action
        if risk_score > 60:
            action = "Legal notice"
        elif risk_score > 30:
            action = "Phone call"
        else:
            action = "SMS reminder"
            
        return {
            "probability": round(prob, 4),
            "risk_score": risk_score,
            "suggested_recovery_action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
