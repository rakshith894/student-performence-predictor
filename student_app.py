import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [2,4,6,8,10],
    'Attendance': [60,70,80,90,95],
    'Score': [30,50,65,85,90]
}
df = pd.DataFrame(data)

X = df[['Hours', 'Attendance']]
y = df['Score']
model = LinearRegression()
model.fit(X, y)

st.title("🎓 Student Performance Predictor")

hours = st.number_input("Enter study hours:", min_value=0, max_value=12, value=6)
attendance = st.number_input("Enter attendance %:", min_value=0, max_value=100, value=80)

if st.button("Predict Score"):
    prediction = model.predict([[hours, attendance]])[0]
    st.success(f"Predicted Score: {prediction:.2f}")
