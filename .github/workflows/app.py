import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the Tourism Package Prediction model from Hugging Face
model_path = hf_hub_download(
    repo_id="namita2025/Pacakage_Prediction_Model",
    filename="Pacakage_Prediction_Model.pkl"
)
model = joblib.load(model_path)

st.title("Tourism Package Prediction Model App")
st.write("""
This app predicts whether a customer is likely to purchase a tourism package.
Enter the customer's details below:
""")

# User input fields based on your dataset
CustomerID = st.text_input("Customer ID (optional)")
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Profile", "Advertisement"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=5)
NumberOfPersonVisiting = st.number_input("NumberOf Person Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=20, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1,2,3,4,5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=20, value=2)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income (in $)", min_value=0, max_value=100000, value=5000)

# Assemble the inputs into a DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])

# Prediction button
if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
