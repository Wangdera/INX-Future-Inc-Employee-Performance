import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load Model, Scaler, and Encoders
with open("best_xgb.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Mapping for ordinal categorical variables
category_mappings = {
    "EmpEnvironmentSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}, 
    "EmpJobInvolvement": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
    "EmpJobSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}, 
    "PerformanceRating": {"Low": 1, "Good": 2, "Excellent": 3, "Outstanding": 4}, 
    "EmpRelationshipSatisfaction": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}, 
    "EmpWorkLifeBalance": {"Bad": 1, "Good": 2, "Better": 3, "Best": 4}, 
    "EmpEducationLevel": {"Below College": 1, "College": 2, "Bachelor": 3, "Master": 4, "Doctor": 5} 
}

# Streamlit App Title
st.title("INX Future Inc. Employee Performance Prediction")
st.write("Enter employee details to predict performance.")

# Input Fields for Features
Age = st.number_input("Age", min_value=18, max_value=65, value=30)
Gender = st.selectbox("Gender", ["Male", "Female"])
EducationBackground = st.selectbox("Education Background", ["Marketing", "Life Sciences", "Human Resources", "Medical", "Other", "Technical Degree"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
EmpDepartment = st.selectbox("Employee Department", ["Human Resources", "Sales", "Development", "Data Science", "Finance", "Research & Development"])
EmpJobRole = st.selectbox("Job Role", ["Sales Executive", "Manager", "Developer", "Sales Representative", "Human Resources", "Senior Developer", "Data Scientist"])
BusinessTravelFrequency = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
DistanceFromHome = st.number_input("Distance From Home", min_value=1, max_value=29, value=2)
NumCompaniesWorked = st.number_input("No. Of Companies Worked", min_value=0, max_value=9, value=1)
EmpEducationLevel = st.selectbox("Employee Education Level", ["Below College", "College", "Bachelor", "Master", "Doctor"])
EmpEnvironmentSatisfaction = st.selectbox("Employee Environment Satisfaction", ["Low", "Medium", "High", "Very High"])
EmpHourlyRate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=50)
EmpJobInvolvement = st.selectbox("Employee Job Involvement", ["Low", "Medium", "High", "Very High"])
EmpJobLevel = st.number_input("Job Level", min_value=1, max_value=5, value=3)
EmpJobSatisfaction = st.selectbox("Employee Job Satisfaction", ["Low", "Medium", "High", "Very High"])
OverTime = st.selectbox("Overtime", ["Yes", "No"])
EmpLastSalaryHikePercent = st.slider("Last Salary Hike (%)", min_value=5, max_value=25, value=10)
EmpRelationshipSatisfaction = st.selectbox("Employee Relationship Satisfaction", ["Low", "Medium", "High", "Very High"])
TotalWorkExperienceInYears = st.slider("Total Work Experience (Yrs)", min_value=0, max_value=40, value=1)
TrainingTimesLastYear = st.slider("Training Times Last Year", min_value=0, max_value=6, value=3)
EmpWorkLifeBalance = st.selectbox("Work Life Balance", ["Bad", "Good", "Better", "Best"])
ExperienceYearsAtThisCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
ExperienceYearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, max_value=20, value=5)
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=3)
YearsWithCurrManager = st.number_input("Years With Current Manager", min_value=0, max_value=17, value=5)
Attrition = st.selectbox("Attrition", ["Yes", "No"])

# Convert input into DataFrame
input_data = pd.DataFrame({
    "Age": [Age], 
    "Gender": [Gender], 
    "EducationBackground": [EducationBackground],
    "MaritalStatus": [MaritalStatus], 
    "EmpDepartment": [EmpDepartment], 
    "EmpJobRole": [EmpJobRole],   
    "BusinessTravelFrequency": [BusinessTravelFrequency], 
    "DistanceFromHome": [DistanceFromHome],  
    "NumCompaniesWorked": [NumCompaniesWorked], 
    "EmpEducationLevel": [EmpEducationLevel],  
    "EmpEnvironmentSatisfaction": [EmpEnvironmentSatisfaction], 
    "EmpHourlyRate": [EmpHourlyRate],  
    "EmpJobInvolvement": [EmpJobInvolvement], 
    "EmpJobLevel": [EmpJobLevel],  
    "EmpJobSatisfaction": [EmpJobSatisfaction], 
    "OverTime": [OverTime],  
    "EmpLastSalaryHikePercent": [EmpLastSalaryHikePercent],  
    "EmpRelationshipSatisfaction": [EmpRelationshipSatisfaction], 
    "TotalWorkExperienceInYears": [TotalWorkExperienceInYears],  
    "TrainingTimesLastYear": [TrainingTimesLastYear], 
    "EmpWorkLifeBalance": [EmpWorkLifeBalance],  
    "ExperienceYearsAtThisCompany": [ExperienceYearsAtThisCompany], 
    "ExperienceYearsInCurrentRole": [ExperienceYearsInCurrentRole], 
    "YearsSinceLastPromotion": [YearsSinceLastPromotion], 
    "YearsWithCurrManager": [YearsWithCurrManager],  
    "Attrition": [Attrition]  
})

# Apply mappings for ordinal categorical features
for column, mapping in category_mappings.items():
    if column in input_data.columns:
        input_data[column] = input_data[column].map(mapping)

# Encode remaining categorical features using LabelEncoder
categorical_features = [
    "Gender", "EducationBackground", "MaritalStatus", "EmpDepartment", 
    "EmpJobRole", "BusinessTravelFrequency", "OverTime", "Attrition", "EmpEducationLevel"
]

for col in categorical_features:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
    else:
        st.error(f"Label encoder for '{col}' is missing.")
        st.stop()

# Convert encoded categorical columns to integers
input_data[categorical_features] = input_data[categorical_features].astype(int)

# Scale numerical features
numerical_features = [
    "Age", "DistanceFromHome", "EmpHourlyRate", "EmpJobLevel", "NumCompaniesWorked", 
    "EmpLastSalaryHikePercent", "TotalWorkExperienceInYears", "TrainingTimesLastYear", 
    "ExperienceYearsAtThisCompany", "ExperienceYearsInCurrentRole", "YearsSinceLastPromotion", 
    "YearsWithCurrManager"
]
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Ensure feature order matches model input
input_data = input_data[model.feature_names_in_]

# Predict Performance
if st.button("Predict Performance"):
    prediction = model.predict(input_data)
    st.write(f"Raw Model Prediction: {prediction[0]}")  # Debugging

    performance_rating_map = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}
    performance_label = performance_rating_map.get(prediction[0], "Unknown")
    
    st.success(f"Predicted Performance: {performance_label}")

