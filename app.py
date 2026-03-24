import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 5000

    data = {
        "transaction_id": np.arange(1, n+1),
        "user_id": np.random.randint(1000, 1100, n),
        "amount": np.random.randint(10, 50000, n),
        "hour": np.random.randint(0, 24, n),
        "is_new_recipient": np.random.randint(0, 2, n),
        "transaction_count_1hr": np.random.randint(1, 10, n),
    }

    df = pd.DataFrame(data)

    # Fraud logic (your innovation)
    df["is_fraud"] = (
        ((df["amount"] > 20000) & (df["is_new_recipient"] == 1)) |
        (df["transaction_count_1hr"] > 5) |
        (df["hour"].isin([0,1,2,3]))
    ).astype(int)

    # Save dataset (for submission)
    df.to_csv("upi_fraud_dataset.csv", index=False)

    return df

df = generate_data()

@st.cache_resource
def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model

model = train_model(df)


st.title("💳 UPI Fraud Detection System")

st.write("Enter transaction details:")

amount = st.number_input("Transaction Amount", min_value=1)
hour = st.slider("Transaction Hour (0-23)", 0, 23)
is_new_recipient = st.selectbox("New Recipient?", [0, 1])
txn_count = st.slider("Transactions in Last Hour", 1, 10)

# Dummy IDs (not important for prediction)
transaction_id = 9999
user_id = 1001
if st.button("Check Fraud"):
    input_data = pd.DataFrame([{
        "transaction_id": transaction_id,
        "user_id": user_id,
        "amount": amount,
        "hour": hour,
        "is_new_recipient": is_new_recipient,
        "transaction_count_1hr": txn_count
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Safe Transaction. Probability: {probability:.2f}")
st.subheader("📊 Sample Dataset Preview")
st.dataframe(df.head())
