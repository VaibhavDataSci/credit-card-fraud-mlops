"""Simple banking-facing frontend for the existing fraud prediction API."""

import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000")
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(
    page_title="Transaction Fraud Check",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f9fc; color: #172033; }
    .block-container { max-width: 920px; padding: 2.5rem 1.2rem 4rem; }
    .hero { background: #ffffff; border: 1px solid #e5eaf2; border-radius: 18px; padding: 2rem 2.2rem; box-shadow: 0 8px 28px rgba(24, 42, 73, .06); }
    .hero h1 { color: #172033; font-size: 2.15rem; margin: 0 0 .4rem; letter-spacing: -.025em; }
    .hero p { color: #60708a; font-size: 1.05rem; margin: 0; }
    .section-title { color: #172033; font-size: 1.25rem; font-weight: 750; margin: 1.8rem 0 .8rem; }
    div[data-testid="stForm"] { background: #ffffff; border: 1px solid #e5eaf2; border-radius: 16px; padding: 1.35rem 1.45rem .65rem; box-shadow: 0 8px 24px rgba(24, 42, 73, .04); }
    div[data-testid="stForm"] label p { color: #24324a; font-weight: 650; }
    .helper { color: #738199; font-size: .88rem; margin-top: -.25rem; margin-bottom: .9rem; }
    .result-card { border-radius: 16px; padding: 1.55rem 1.65rem; margin-top: 1rem; border: 1px solid; box-shadow: 0 8px 24px rgba(24, 42, 73, .05); }
    .result-fraud { background: #fff7f5; border-color: #f4c5bd; }
    .result-safe { background: #f2fbf6; border-color: #b8e4ca; }
    .result-heading { color: #172033; font-size: 1.05rem; font-weight: 700; margin-bottom: .75rem; }
    .result-title { font-size: 1.48rem; font-weight: 800; margin-bottom: .95rem; }
    .result-safe .result-title { color: #167044; }
    .result-fraud .result-title { color: #b42318; }
    .risk-label { color: #60708a; font-size: .9rem; font-weight: 650; }
    .risk-value { color: #172033; font-size: 2rem; font-weight: 800; margin: .15rem 0 .7rem; }
    .explanation { color: #4b5c76; font-size: .96rem; line-height: 1.5; margin-top: .9rem; }
    .stButton > button { border-radius: 10px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_preset(values: dict[str, Any]) -> None:
    for key, value in values.items():
        st.session_state[key] = value
    st.session_state.pop("prediction", None)


def request_prediction(payload: dict[str, Any]) -> None:
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15)
        if response.ok:
            st.session_state.prediction = response.json()
        else:
            st.session_state.pop("prediction", None)
            st.error("We could not check this transaction right now. Please try again.")
    except requests.RequestException:
        st.session_state.pop("prediction", None)
        st.error("The transaction checking service is temporarily unavailable. Please try again.")


for key, value in {
    "transaction_type": "TRANSFER",
    "amount": 1000.0,
    "oldbalance_org": 1000.0,
    "newbalance_orig": 0.0,
    "oldbalance_dest": 0.0,
    "newbalance_dest": 1000.0,
}.items():
    st.session_state.setdefault(key, value)

st.markdown(
    """
    <div class="hero">
      <h1>Transaction Fraud Check</h1>
      <p>Review a transaction for potential fraud risk before proceeding.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Transaction information</div>', unsafe_allow_html=True)
preset_columns = st.columns([1, 1, 2])
with preset_columns[0]:
    if st.button("Load Normal Transaction", use_container_width=True):
        load_preset({"transaction_type": "PAYMENT", "amount": 10.0, "oldbalance_org": 1000.0, "newbalance_orig": 990.0, "oldbalance_dest": 0.0, "newbalance_dest": 10.0})
with preset_columns[1]:
    if st.button("Load Suspicious Transaction", use_container_width=True):
        load_preset({"transaction_type": "TRANSFER", "amount": 1000.0, "oldbalance_org": 1000.0, "newbalance_orig": 0.0, "oldbalance_dest": 0.0, "newbalance_dest": 1000.0})

with st.form("transaction_form"):
    transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES, key="transaction_type", help="Select the type of transaction.")
    st.markdown('<div class="helper">Select the type of transaction.</div>', unsafe_allow_html=True)
    amount = st.number_input("Transaction Amount", min_value=0.0, step=10.0, key="amount", help="Enter the amount being transferred.")
    oldbalance_org = st.number_input("Sender's Balance Before Transaction", min_value=0.0, step=10.0, key="oldbalance_org", help="Balance in the sender's account before this transaction.")
    newbalance_orig = st.number_input("Sender's Balance After Transaction", min_value=0.0, step=10.0, key="newbalance_orig", help="Balance in the sender's account after this transaction.")
    oldbalance_dest = st.number_input("Receiver's Balance Before Transaction", min_value=0.0, step=10.0, key="oldbalance_dest", help="Balance in the receiver's account before this transaction.")
    newbalance_dest = st.number_input("Receiver's Balance After Transaction", min_value=0.0, step=10.0, key="newbalance_dest", help="Balance in the receiver's account after this transaction.")
    submitted = st.form_submit_button("Check Transaction", type="primary", use_container_width=True)

if submitted:
    if amount <= 0:
        st.error("Please enter a valid transaction amount.")
    else:
        request_prediction({"type": transaction_type, "amount": amount, "oldbalanceOrg": oldbalance_org, "newbalanceOrig": newbalance_orig, "oldbalanceDest": oldbalance_dest, "newbalanceDest": newbalance_dest})

prediction = st.session_state.get("prediction")
if prediction:
    is_fraud = prediction["is_fraud"]
    probability = prediction["fraud_probability"]
    result_class = "result-fraud" if is_fraud else "result-safe"
    result_title = "⚠ Potential fraud detected" if is_fraud else "✓ Transaction appears legitimate"
    explanation = "This transaction has characteristics associated with potentially fraudulent activity. Please review the transaction before proceeding." if is_fraud else "No immediate fraud risk was identified for this transaction."
    st.markdown('<div class="section-title">Transaction result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-card {result_class}"><div class="result-heading">Transaction Result</div><div class="result-title">{result_title}</div><div class="risk-label">Fraud risk</div><div class="risk-value">{probability:.0%}</div></div>', unsafe_allow_html=True)
    st.progress(float(probability), text=f"Fraud risk · {probability:.0%}")
    st.markdown(f'<div class="explanation">{explanation}</div>', unsafe_allow_html=True)
