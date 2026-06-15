import streamlit as st
import pickle
import numpy as np

# Load models
model_breakfast = pickle.load(open("models/breakfast.pkl", "rb"))
model_lunch = pickle.load(open("models/lunch.pkl", "rb"))
model_dinner = pickle.load(open("models/dinner.pkl", "rb"))

# Load encoders
le_gender = pickle.load(open("models/gender.pkl", "rb"))
le_activity = pickle.load(open("models/activity.pkl", "rb"))
le_disease = pickle.load(open("models/disease.pkl", "rb"))
le_bmi = pickle.load(open("models/bmi.pkl", "rb"))

# Safe transform function
def safe_transform(encoder, value):
    value = value.strip().lower()
    if value not in encoder.classes_:
        return encoder.transform([encoder.classes_[0]])[0]
    return encoder.transform([value])[0]

st.title("🍎 Smart Food Recommendation System")

st.write("Enter Your Details")

# Inputs
age = st.number_input("Age", 10, 100)
weight = st.number_input("Weight (kg)", 30, 150)
height = st.number_input("Height (cm)", 120, 210)

gender = st.selectbox("Gender", ["Male","Female"])
activity = st.selectbox("Activity Level", ["Low","Medium","High"])
disease = st.selectbox("Disease", ["None","BP","Diabetes"])
bmi = st.selectbox("BMI Category", ["Normal","Overweight","Obese"])

if st.button("Recommend Food"):

    gender = safe_transform(le_gender, gender)
    activity = safe_transform(le_activity, activity)
    disease = safe_transform(le_disease, disease)
    bmi = safe_transform(le_bmi, bmi)

    input_data = np.array([[age, weight, height, gender, activity, disease, bmi]])

    breakfast = model_breakfast.predict(input_data)
    lunch = model_lunch.predict(input_data)
    dinner = model_dinner.predict(input_data)

    st.success("🍽 Recommended Diet Plan")

    st.write("🍳 Breakfast:", breakfast[0])
    st.write("🍛 Lunch:", lunch[0])
    st.write("🥗 Dinner:", dinner[0])