"""Pydantic schemas for the fraud inference API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Raw transaction fields required to build the trained model features."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "TRANSFER",
                "amount": 1000.0,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1000.0,
            }
        }
    )

    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Prediction and deployment context returned by the API."""

    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
    threshold: float
    model_name: str
    model_alias: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    alias: str
    model_uri: str
    version: str | None = None
    run_id: str | None = None
    strategy: str | None = None
    threshold: float
