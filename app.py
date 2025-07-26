import streamlit as st
import pandas as pd
import pickle

st.title("Diabetes Prediction")
st.caption("You can predict patients having diabetes or no diabetes by passing some values")
st.divider

df = pd.read_csv(r"C:/Users/DELL/class/Project/diabetes.csv")

preg = st.number_input(label='Preganancies')
gluc= st.number_input(label='Glucose')
bp= st.number_input(label='BloodPressure')
sta=st.number_input(label='SkinThickness')
ins=st.number_input(label='Insulin')
bmi=st.number_input(label='BMI')
dpf=st.number_input(label='DiabetesPedigreeFunction')
age=st.number_input(label='Age')

input_array = [[preg, gluc, bp, sta, ins, bmi, dpf, age]]

# Model Initialization
@st.cache_resource
def get_model():
    with open("rf_model.pkl","rb") as model_data:
        rf_model = pickle.load(model_data)
    return rf_model

button_state = st.button("Submit & Predict")
if button_state == True:
    rf_model = get_model()
    st.write(f"Input given by the user is: {input_array}")
    result = rf_model.predict(input_array)
    if result == 1:
        st.info("Pediction: Diabetes Positive")
    else:
        st.balloons()
        st.info("Pediction: Diabetes Negative")