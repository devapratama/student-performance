import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
preprocessor = joblib.load('preprocessor_model.joblib')
model = joblib.load('best_rf_model.joblib')

def predict_student_status(input_data):
    """
    Function to predict the student status and return results.
    """
    try:
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)
        prediction_prob = model.predict_proba(processed_data)
        return prediction, prediction_prob
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Streamlit App
st.title("Student Status Prediction")
st.write("Predict whether a student will graduate, dropout, or remain enrolled.")

# Input Form
with st.form(key="input_form"):
    st.header("Enter Student Details")

    # Numeric Inputs
    age_at_enrollment = st.number_input("Age at Enrollment", min_value=16, max_value=100, value=20)
    previous_qualification_grade = st.number_input("Previous Qualification Grade (0-200)", min_value=0, max_value=200, value=100)
    admission_grade = st.number_input("Admission Grade (0-200)", min_value=0, max_value=200, value=150)

    # Credits and Grades for 1st Semester
    curricular_units_1st_sem_credited = st.number_input("Credits Earned in 1st Semester", min_value=0, max_value=100, value=5)
    curricular_units_1st_sem_enrolled = st.number_input("Courses Enrolled in 1st Semester", min_value=0, max_value=50, value=5)
    curricular_units_1st_sem_evaluations = st.number_input("Evaluations in 1st Semester", min_value=0, max_value=50, value=5)
    curricular_units_1st_sem_approved = st.number_input("Courses Approved in 1st Semester", min_value=0, max_value=50, value=4)
    curricular_units_1st_sem_grade = st.number_input("Average Grade in 1st Semester (0-20)", min_value=0.0, max_value=20.0, value=15.0, step=0.1)
    curricular_units_1st_sem_without_evaluations = st.number_input("Courses Without Evaluations in 1st Semester", min_value=0, max_value=10, value=0)

    # Credits and Grades for 2nd Semester
    curricular_units_2nd_sem_credited = st.number_input("Credits Earned in 2nd Semester", min_value=0, max_value=100, value=5)
    curricular_units_2nd_sem_enrolled = st.number_input("Courses Enrolled in 2nd Semester", min_value=0, max_value=50, value=5)
    curricular_units_2nd_sem_evaluations = st.number_input("Evaluations in 2nd Semester", min_value=0, max_value=50, value=5)
    curricular_units_2nd_sem_approved = st.number_input("Courses Approved in 2nd Semester", min_value=0, max_value=50, value=4)
    curricular_units_2nd_sem_grade = st.number_input("Average Grade in 2nd Semester (0-20)", min_value=0.0, max_value=20.0, value=15.0, step=0.1)
    curricular_units_2nd_sem_without_evaluations = st.number_input("Courses Without Evaluations in 2nd Semester", min_value=0, max_value=10, value=0)

    # Categorical Inputs
    marital_status = st.selectbox("Marital Status", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: [
        "Single", "Married", "Widower", "Divorced", "De facto Union", "Legally Separated"
    ][x-1])
    application_mode = st.selectbox("Application Mode", options=[1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57], format_func=lambda x: {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores Island)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira Island)",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
        27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"
    }.get(x, "Unknown"))
    application_order = st.selectbox("Application Order", options=range(0, 10), format_func=lambda x: f"Choice {x}")
    course = st.selectbox("Course", options=[33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991], format_func=lambda x: {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)"
    }.get(x, "Unknown"))
    daytime_evening_attendance = st.selectbox("Attendance Type", options=[1, 0], format_func=lambda x: ["Daytime", "Evening"][x])
    previous_qualification = st.selectbox("Previous Qualification", options=[1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43], format_func=lambda x: {
        1: "Secondary education",
        2: "Higher education - bachelor's degree",
        3: "Higher education - degree",
        4: "Higher education - master's",
        5: "Higher education - doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling - not completed",
        10: "11th year of schooling - not completed",
        12: "Other - 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling - not completed",
        19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        39: "Technological specialization course",
        40: "Higher education - degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education - master (2nd cycle)"
    }.get(x, "Unknown"))
    nationality = st.selectbox("Nationality", options=[1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109], format_func=lambda x: {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 
        14: "English", 17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 
        25: "Mozambican", 26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian", 
        100: "Moldova (Republic of)", 101: "Mexican", 103: "Ukrainian", 105: "Russian", 
        108: "Cuban", 109: "Colombian"
    }.get(x, "Unknown"))
    
    displaced = st.selectbox("Displaced?", options=[1, 0], format_func=lambda x: ["Yes", "No"][x])
    educational_special_needs = st.selectbox("Educational Special Needs?", options=[1, 0], format_func=lambda x: ["Yes", "No"][x])
    debtor = st.selectbox("Debtor?", options=[1, 0], format_func=lambda x: ["Yes", "No"][x])
    tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date?", options=[1, 0], format_func=lambda x: ["Yes", "No"][x])
    gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: ["Male", "Female"][x])
    scholarship_holder = st.selectbox("Scholarship Holder?", options=[1, 0], format_func=lambda x: ["Yes", "No"][x])
    international = st.selectbox("International Student?", options=[1, 0], format_func=lambda x: ["Yes", "No"][x])

    # Economic Indicators
    unemployment_rate = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    gdp = st.number_input("GDP Growth Rate (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)

    submit_button = st.form_submit_button(label="Predict")

# Prediction Logic
if submit_button:
    input_data = pd.DataFrame({
        "Marital_status": [marital_status],
        "Application_mode": [application_mode],
        "Application_order": [application_order],
        "Course": [course],
        "Daytime_evening_attendance": [daytime_evening_attendance],
        "Previous_qualification": [previous_qualification],
        "Previous_qualification_grade": [previous_qualification_grade],
        "Nacionality": [nationality],
        "Admission_grade": [admission_grade],
        "Displaced": [displaced],
        "Educational_special_needs": [educational_special_needs],
        "Debtor": [debtor],
        "Tuition_fees_up_to_date": [tuition_fees_up_to_date],
        "Gender": [gender],
        "Scholarship_holder": [scholarship_holder],
        "Age_at_enrollment": [age_at_enrollment],
        "International": [international],
        "Curricular_units_1st_sem_credited": [curricular_units_1st_sem_credited],
        "Curricular_units_1st_sem_enrolled": [curricular_units_1st_sem_enrolled],
        "Curricular_units_1st_sem_evaluations": [curricular_units_1st_sem_evaluations],
        "Curricular_units_1st_sem_approved": [curricular_units_1st_sem_approved],
        "Curricular_units_1st_sem_grade": [curricular_units_1st_sem_grade],
        "Curricular_units_1st_sem_without_evaluations": [curricular_units_1st_sem_without_evaluations],
        "Curricular_units_2nd_sem_credited": [curricular_units_2nd_sem_credited],
        "Curricular_units_2nd_sem_enrolled": [curricular_units_2nd_sem_enrolled],
        "Curricular_units_2nd_sem_evaluations": [curricular_units_2nd_sem_evaluations],
        "Curricular_units_2nd_sem_approved": [curricular_units_2nd_sem_approved],
        "Curricular_units_2nd_sem_grade": [curricular_units_2nd_sem_grade],
        "Curricular_units_2nd_sem_without_evaluations": [curricular_units_2nd_sem_without_evaluations],
        "Unemployment_rate": [unemployment_rate],
        "Inflation_rate": [inflation_rate],
        "GDP": [gdp]
    })

    prediction, prediction_prob = predict_student_status(input_data)
    if prediction is not None:
        status_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        
        if isinstance(prediction[0], int):
            predicted_status = status_map.get(prediction[0], "Unrecognized Status")
        else:
            predicted_status = prediction[0]
        
        probabilities = {status_map[i]: round(prob, 2) for i, prob in enumerate(prediction_prob[0])}
        st.subheader("Prediction Result")
        st.write(f"Predicted Status: **{predicted_status}**")
        st.write("Probabilities:")
        for status, prob in probabilities.items():
            st.write(f"- {status}: **{prob * 100:.2f}%**")