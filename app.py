import pickle
import streamlit as st
import numpy as np

# Load model and scaler
loaded_model = pickle.load(open("Model.sav", "rb"))
loaded_scaler = pickle.load(open("Scaler.sav", "rb"))

def diabete_prediction(user_input):
    input_data_as_np = np.asarray(user_input, dtype=float).reshape(1, -1)
    std_data = loaded_scaler.transform(input_data_as_np)
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return (
        
            "✅ **Prediction**: The individual is **non-diabetic**.\n\n"
            "### 🩺 Health Advice:\n"
            "- Maintain a balanced diet rich in fiber, vegetables, lean proteins, and whole grains.\n"
            "- Exercise regularly (30 mins of brisk walking, 5 times/week).\n"
            "- Annual blood sugar checkups are recommended.\n"
            "- Stay hydrated and maintain a healthy weight."
        )
    else:
        return (
            "⚠️ **Prediction**: The individual is **likely diabetic**.\n\n"
            "### 🩺 Medical Advice:\n"
            "- Please consult a certified doctor.\n"
            "- Medication such as Metformin may be needed.\n"
            "- Regular monitoring of glucose, A1C, and cholesterol.\n\n"
            "### 🥗 Diet & Lifestyle:\n"
            "- Avoid sugary foods and refined carbs.\n"
            "- Include oats, legumes, green veggies.\n"
            "- Drink water, avoid smoking/alcohol.\n\n"
            "### 🏃 Exercise Plan:\n"
            "- 150 minutes of moderate-intensity activity/week.\n"
            "- Include walking, yoga, cycling, or swimming.\n"
            "- Add strength training twice a week."
        )

def main():
    #sidebar text color and background
    st.markdown("""
    <style>
        /* Change sidebar background and text color */
        [data-testid="stSidebar"] {
            background-color: #003366;  /* Dark Blue */
            color: white;
        }

        /* Change sidebar title and content text to white */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] ul,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] a {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

    # Optional sidebar image
    st.sidebar.image("what-is-machine-learning.jpg", width=250)
    st.markdown("""
        <style>
            .stApp {
                background-color: #C70039;
            }
        </style>
    """, unsafe_allow_html=True)

     # Sidebar
    st.sidebar.title("How This App Works")
    st.sidebar.markdown("""
This app uses a **machine learning model** to predict whether a person is likely to have diabetes.

### 📥 Step-by-Step:
1. **Enter health parameters** like glucose, BMI, age, etc.
2. The app applies the same **standard scaling** as used during training.
3. It uses a **trained model** to make a prediction.
4. You get personalized **medical and lifestyle advice** based on the result.

### 🔍 Features:
- Trained on real medical data
- Fast and accurate prediction
- Gives health & care suggestions
""")



    # Main app title and description
    st.title("🩺 Diabetes Prediction Web App")
    st.write("Enter health parameters to check diabetes likelihood:")

    # Input fields
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    # Prediction logic
    if st.button("Predict Diabetes"):
        if '' in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
            st.error("⚠️ Please fill all the fields before predicting.")
        else:
            diagnose = diabete_prediction([
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
            ])
            st.markdown(diagnose)

    # Footer
    st.markdown("""
    <hr style="margin-top: 50px;"/>
    <div style="text-align: center; padding: 10px 0; color: white; font-size: 16px;">
        © 2025 <strong>Lalit</strong> | Built with ❤️ using 
        <a href="https://streamlit.io" target="_blank" style="color: white;">Streamlit</a>
    </div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
