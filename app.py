import streamlit as st
import pickle
import pandas as pd

# Load models and feature columns
reg_model = pickle.load(open("model/regression_model.pkl", "rb"))
clf_model = pickle.load(open("model/classification_model.pkl", "rb"))
feature_columns = pickle.load(open("model/feature_columns.pkl", "rb"))

st.title("🎓 Student Performance Analyzer")

st.sidebar.header("Input Student Details")

age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Study Time (1-4)", 1, 4, 2)
failures = st.sidebar.slider("Past Failures", 0, 3, 0)
absences = st.sidebar.slider("Absences", 0, 75, 5)

# Create base dictionary with all features = 0
input_dict = {col: 0 for col in feature_columns}

# Fill selected inputs
input_dict["age"] = age
input_dict["studytime"] = studytime
input_dict["failures"] = failures
input_dict["absences"] = absences

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

if st.button("Predict"):

    grade_prediction = reg_model.predict(input_df)[0]
    pass_prediction = clf_model.predict(input_df)[0]

    st.subheader("📊 Results")

    st.write(f"Predicted Final Grade: {grade_prediction:.2f}")

    if pass_prediction == 1:
        st.success("Status: Likely PASS ✅")
    else:
        st.error("Status: At Risk (FAIL) ⚠️")

# I implemented both regression and classification models, and observed occasional disagreement due to different learning objectives.