import streamlit as st
import joblib as jb
import numpy as np
# Page config
st.set_page_config(
    page_title="ML based Employee Attrition Prediction",
    page_icon="🤖",
    layout="centered"
)

# Sidebar
st.sidebar.title("⚙ Model Settings")
st.sidebar.info("Select a machine learning model and run predictions.")

# Main title
st.markdown("<h1 style='text-align: center;'>🤖 ML based Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Interactive Streamlit Application</h4>", unsafe_allow_html=True)
st.divider()

# Welcome section
st.markdown("### 👋 Welcome")
st.write("This application allows you to select and run different machine learning models for predicting Employee Attrition.")

# Model selection section
st.markdown("### 🔍 Choose Your Model")
#Models mapping
model_files={
    "Logistic Regression":"models\\log_reg_model.pkl",
    "Decision Tree":"models\\dt_model.pkl",
    "Random Forest":"models\\ rf_model.pkl",
    "Neural Networks":"models\\ mlp_model.pkl"}
#model selection
models = st.selectbox(
    "Select a Machine Learning Model:", list(model_files.keys()))
#Loading the model
model_path=model_files[models]
selected_model=jb.load(model_path)
le_department = jb.load("models/le_department.pkl")
le_salary = jb.load("models/le_salary.pkl")
st.success(f"✅ You loaded: **{models}**")


st.subheader("Enter Employee Details")
sat_level=st.number_input("Employee satisfaction level in the company",min_value=0.0, max_value=1.0)
evl=st.number_input("Employee last evaluation score", min_value=0.0, max_value=1.0)
proj_done=st.number_input("Number of projects done by the Employee", max_value=10)
work_HRS=st.number_input("Average monthly working hours of the Employee", max_value=730)
total_time=st.number_input("Number of years spent in the company", max_value=12)
accidents=st.number_input("Did the employee have any work related accidents. 1 means yes, 0 means no.",min_value=0, max_value=1)
promotion=st.number_input("Was the employee promoted in the last 5 years. 1 means yes, 0 means no.",min_value=0, max_value=1)
department=st.selectbox("Select Department", le_department.classes_)
salary = st.selectbox("Salary Level", le_salary.classes_)

department_encoded = le_department.transform([department])[0]
salary_encoded = le_salary.transform([salary])[0]
# Run button
input_data = np.array([[sat_level, evl, proj_done,
                            work_HRS, total_time,
                            accidents, promotion,
                            department_encoded, salary_encoded]])
if st.button("🚀 Run Model"):
    st.info(f"Running **{models}** model...")
    prediction=selected_model.predict(input_data)
    st.success("🎉 Model executed successfully!")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by Mahmood Ali Khan | Streamlit ML App</p>",
    unsafe_allow_html=True
)