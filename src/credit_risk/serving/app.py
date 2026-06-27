import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.credit_risk.features.engineer import CreditRiskFeatureEngineer

MODEL_PATH = Path(__file__).resolve().parents[4] / "outputs" / "best_model_bundle.pkl"

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts the probability that a loan applicant will default.",
    version="1.0.0",
)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

_engineer    = CreditRiskFeatureEngineer()
_selector    = bundle["selector"]
_preprocessor = bundle["preprocessor"]
_model       = bundle["model"]


class LoanApplication(BaseModel):
    # Core financial ratios — the most important inputs for credit scoring
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income in local currency")
    AMT_CREDIT: float       = Field(..., gt=0, description="Loan amount requested")
    AMT_ANNUITY: float      = Field(..., gt=0, description="Monthly loan repayment amount")
    AMT_GOODS_PRICE: float  = Field(..., gt=0, description="Price of the goods the loan is for")

    DAYS_BIRTH: int          = Field(..., lt=0, description="Days since birth (negative number)")
    DAYS_EMPLOYED: int       = Field(..., description="Days since employment started (negative = employed, 365243 = unemployed)")
    CNT_FAM_MEMBERS: float   = Field(..., ge=1, description="Number of family members")

    NAME_EDUCATION_TYPE: str = Field(..., description="Highest education level")
    NAME_INCOME_TYPE: str    = Field(..., description="Income source type")
    ORGANIZATION_TYPE: str   = Field(..., description="Type of employer organisation")
    OCCUPATION_TYPE: str     = Field(default="Unknown", description="Occupation type")

    EXT_SOURCE_1: float = Field(default=0.5, ge=0, le=1, description="External credit score 1 (0-1)")
    EXT_SOURCE_2: float = Field(default=0.5, ge=0, le=1, description="External credit score 2 (0-1)")
    EXT_SOURCE_3: float = Field(default=0.5, ge=0, le=1, description="External credit score 3 (0-1)")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_grade: str
    decision: str
    model_version: str


def _grade(prob: float) -> tuple[str, str]:
    """Map a probability to a risk grade and a lending decision."""
    if prob < 0.05:
        return "A", "Approve"
    elif prob < 0.10:
        return "B", "Approve"
    elif prob < 0.20:
        return "C", "Review"
    elif prob < 0.35:
        return "D", "Review"
    else:
        return "E", "Decline"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "LightGBM + Optuna credit risk pipeline",
        "test_roc_auc": bundle.get("test_roc_auc"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    df = pd.DataFrame([application.model_dump()])

    try:
        df = _engineer.transform(df)
        df_sel  = _selector.transform(df)
        df_proc = _preprocessor.transform(df_sel)
        prob    = float(_model.predict_proba(df_proc)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    grade, decision = _grade(prob)

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_grade=grade,
        decision=decision,
        model_version=f"lgb-optuna-auc{bundle.get('test_roc_auc', 'n/a')}",
    )
