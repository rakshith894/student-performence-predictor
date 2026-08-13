import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample training data
data = {
    "study_hours": [2, 4, 6, 8, 10],
    "attendance": [50, 60, 70, 80, 90],
    "pass_fail": [0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# Features and target
X = df[["study_hours", "attendance"]]
y = df["pass_fail"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("Student Pass/Fail Predictor")

study_hours = st.number_input("Enter study hours", min_value=0)
attendance = st.number_input("Enter attendance %", min_value=0, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[study_hours, attendance]])
    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
    st.success(f"Prediction: {result}")
