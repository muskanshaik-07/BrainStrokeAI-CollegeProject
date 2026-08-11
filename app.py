import streamlit as st
import numpy as np
import pandas as pd
import joblib
from google import genai
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn

from pathlib import Path

from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Brain Stroke AI Agent",
    page_icon="🧠",
    layout="wide"
)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
encoders = joblib.load(BASE_DIR / "models" / "encoders.pkl")

dnn_model = load_model(
    BASE_DIR / "models" / "dnn_model.keras"
)

tabnet_model = TabNetClassifier()
tabnet_model.load_model(
    str(BASE_DIR / "models" / "tabnet_model.zip")
)


# -------------------------
# FT Transformer
# -------------------------

class FTTransformer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.embedding = nn.Linear(input_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)

        return torch.sigmoid(x)

ft_model = FTTransformer(10)

ft_model.load_state_dict(
    torch.load(
        BASE_DIR / "models" / "ft_transformer.pth",
        map_location=torch.device("cpu")
    )
)

ft_model.eval()

# Sidebar
# -------------------------

st.sidebar.title("🧠 Brain Stroke AI Agent")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Stroke Prediction",
        "🤖 Clinical Decision Support",
        "💬 AI Health Assistant",
        "📈 Model Comparison",
        "ℹ️ About"
    ]
)
# -------------------------
# HOME PAGE
# -------------------------

if page == "🏠 Home":

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f172a, #1e3a8a);
            padding: 40px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
        ">
            <h1>🧠 Brain Stroke AI Agent</h1>
            <p style="font-size: 20px;">
                AI-powered stroke risk prediction using Electronic Health Records
            </p>
            <p>
                Deep Learning • TabNet • FT-Transformer • Gemini AI
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Welcome 👋")

    st.write(
        """
        This AI Agent analyzes patient health information and predicts
        potential stroke risk using three Machine Learning models.
        """
    )

    st.markdown("### 🤖 AI Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 🧠 DNN
            Deep Neural Network for stroke risk prediction.
            """
        )

    with col2:
        st.markdown(
            """
            ### 📊 TabNet
            Attention-based model designed for tabular healthcare data.
            """
        )

    with col3:
        st.markdown(
            """
            ### 🔬 FT-Transformer
            Transformer-based model for structured patient data.
            """
        )

    st.markdown("---")

    st.markdown("### 💡 What can you do?")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
            **📊 Stroke Prediction**

            Enter patient information and compare predictions
            from DNN, TabNet, and FT-Transformer.
            """
        )

    with col2:
        st.info(
            """
            **💬 AI Health Assistant**

            Ask Gemini questions about stroke, symptoms,
            prevention, and general health information.
            """
        )

    st.warning(
        "⚠️ This application is for educational and decision-support "
        "purposes only. It is not a substitute for professional medical advice."
    )
# -------------------------
# STROKE PREDICTION
# -------------------------

elif page == "📊 Stroke Prediction":

    st.title("📊 Stroke Risk Prediction")

    st.markdown(
        "Enter the patient's health information to estimate stroke risk."
    )

    st.markdown("---")

    # -------------------------
    # Patient Information
    # -------------------------

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        married = st.selectbox(
            "Ever Married",
            ["Yes", "No"]
        )

    with col2:

        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )

        residence = st.selectbox(
            "Residence Type",
            ["Urban", "Rural"]
        )

        glucose = st.number_input(
            "Average Glucose Level",
            min_value=50.0,
            max_value=300.0,
            value=100.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

        smoking = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ]
        )

    st.markdown("---")

    # -------------------------
    # Model Selection
    # -------------------------

    st.subheader("🤖 Select Prediction Model")

    model_choice = st.selectbox(
        "Choose a model",
        [
            "DNN",
            "TabNet",
            "FT-Transformer"
        ]
    )

    st.markdown("")

    if st.button(
        "🔍 Predict Stroke Risk",
        use_container_width=True
    ):

        # Encode categorical values

        gender_encoded = encoders["gender"].transform(
            [gender]
        )[0]

        married_encoded = encoders["ever_married"].transform(
            [married]
        )[0]

        work_type_encoded = encoders["work_type"].transform(
            [work_type]
        )[0]

        residence_encoded = encoders["Residence_type"].transform(
            [residence]
        )[0]

        smoking_encoded = encoders["smoking_status"].transform(
            [smoking]
        )[0]

        # Create input

        input_data = np.array([[
            gender_encoded,
            age,
            hypertension,
            heart_disease,
            married_encoded,
            work_type_encoded,
            residence_encoded,
            glucose,
            bmi,
            smoking_encoded
        ]])

        # Scale

        input_data = scaler.transform(input_data)

        # -------------------------
        # Prediction
        # -------------------------

        if model_choice == "DNN":

            probability = dnn_model.predict(
                input_data,
                verbose=0
            )[0][0]

        elif model_choice == "TabNet":

            probability = tabnet_model.predict_proba(
                input_data
            )[0][1]

        else:

            input_tensor = torch.tensor(
                input_data,
                dtype=torch.float32
            )

            with torch.no_grad():

                probability = ft_model(
                    input_tensor
                ).item()

        prediction = 1 if probability >= 0.5 else 0

        # -------------------------
        # Result
        # -------------------------

        st.markdown("---")

        st.subheader("📋 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.error(
                    "⚠️ HIGH STROKE RISK"
                )

            else:

                st.success(
                    "✅ LOW STROKE RISK"
                )

        with result_col2:

            st.metric(
                "Stroke Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(
            float(probability)
        )

        st.caption(
            f"Prediction generated using {model_choice}."
        )

        st.warning(
            "⚠️ This prediction is for educational and "
            "decision-support purposes only. It is not a medical diagnosis."
        )
elif page == "🤖 Clinical Decision Support":

    st.title("🤖 AI Clinical Decision Support")

    st.markdown(
        "Enter patient information and let Gemini provide an "
        "educational clinical-risk analysis."
    )

    st.info(
        "🩺 This tool provides educational decision support only. "
        "It does not diagnose conditions or replace a healthcare professional."
    )

    st.markdown("---")

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )

    with col2:

        glucose = st.number_input(
            "Average Glucose Level",
            min_value=50.0,
            max_value=300.0,
            value=100.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

        smoking = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ]
        )

        stroke_risk = st.selectbox(
            "Predicted Stroke Risk",
            ["Low", "High"]
        )

    st.markdown("---")

    if st.button(
        "🤖 Generate AI Recommendation",
        use_container_width=True
    ):

        patient_details = f"""
Age: {age}
Gender: {gender}
Hypertension: {hypertension}
Heart Disease: {heart_disease}
Average Glucose Level: {glucose}
BMI: {bmi}
Smoking Status: {smoking}
Predicted Stroke Risk: {stroke_risk}
"""

        prompt = f"""
You are an AI clinical decision support assistant.

Analyze the following patient information:

{patient_details}

Provide:

1. Stroke Risk Summary
2. Possible Risk Factors
3. Lifestyle Recommendations
4. Medical Advice
5. Emergency Warning Signs

Keep the explanation simple and easy to understand.

Important:
This is an educational decision-support system and is not a
replacement for evaluation by a qualified healthcare professional.
Do not diagnose the patient or prescribe medication.
"""

        with st.spinner("🧠 Gemini is analyzing the patient..."):

            try:

                if client is None:
                    raise ValueError(
                        "GEMINI_API_KEY is not configured. Add it in "
                        "Streamlit Cloud → Settings → Secrets (or a local .env)."
                    )

                response = client.models.generate_content(
                    model="gemini-3.6-flash",                    
                    contents=prompt
                )

                st.success("✅ AI Analysis Complete")

                st.markdown("---")

                st.subheader("🧠 Clinical Analysis")

                st.markdown(response.text)

            except Exception as e:

                st.error(
                    f"Gemini Error: {e}"
                )
# -------------------------
# AI HEALTH ASSISTANT
# -------------------------

elif page == "💬 AI Health Assistant":

    st.title("💬 AI Health Assistant")

    st.markdown(
        "Ask questions about stroke, symptoms, prevention, "
        "risk factors, or general health information."
    )

    st.info(
        "🩺 Educational assistant only — it does not diagnose "
        "conditions or replace a healthcare professional."
    )

    # -------------------------
    # Chat History
    # -------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------
    # Chat Input
    # -------------------------

    user_question = st.chat_input(
        "Ask something about stroke or health..."
    )

    if user_question:

        # Show user's message

        with st.chat_message("user"):
            st.markdown(user_question)

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        chat_prompt = f"""
You are an educational AI health assistant.

The user is asking a question about stroke or general health.

User question:
{user_question}

Answer in simple and easy-to-understand language.

You may explain:
- What stroke is
- Stroke symptoms
- Stroke risk factors
- Stroke prevention
- Healthy lifestyle
- General medical terminology

Important safety rules:

- Do not diagnose the user.
- Do not prescribe medication.
- Do not tell the user to start or stop medication.
- If the user describes possible emergency stroke symptoms,
  advise them to seek emergency medical care immediately.
- Clearly explain that your response is educational information.
"""

        with st.chat_message("assistant"):

            with st.spinner("🧠 Thinking..."):

                try:

                    if client is None:
                        raise ValueError(
                            "GEMINI_API_KEY is not configured. Add it in "
                            "Streamlit Cloud → Settings → Secrets (or a local .env)."
                        )

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=chat_prompt
                    )

                    answer = response.text

                    st.markdown(answer)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Gemini Error: {e}"
                    )
# -------------------------
# MODEL COMPARISON
# -------------------------

elif page == "📈 Model Comparison":

    st.title("📈 Model Comparison")

    st.markdown(
        "Compare the performance of the three AI models used "
        "for stroke prediction."
    )

    st.markdown("---")

    # Model performance

    models = [
        "DNN",
        "TabNet",
        "FT-Transformer"
    ]

    accuracies = [
        93.93,
        93.84,
        93.93
    ]

    comparison_data = pd.DataFrame({
        "Model": models,
        "Accuracy (%)": accuracies
    })

    # -------------------------
    # Performance Cards
    # -------------------------

    st.subheader("🏆 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🧠 DNN",
            "93.93%"
        )

    with col2:

        st.metric(
            "📊 TabNet",
            "93.84%"
        )

    with col3:

        st.metric(
            "🔬 FT-Transformer",
            "93.93%"
        )

    st.markdown("---")

    # -------------------------
    # Table
    # -------------------------

    st.subheader("📋 Accuracy Comparison")

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------
    # Chart
    # -------------------------

    st.subheader("📊 Accuracy Visualization")

    chart_data = comparison_data.set_index("Model")

    st.bar_chart(
        chart_data,
        y="Accuracy (%)"
    )

    st.info(
        "DNN and FT-Transformer achieved the highest accuracy "
        "in this model comparison."
    )

    st.warning(
        "⚠️ Accuracy alone does not determine which model is best "
        "for clinical use. Additional metrics such as precision, "
        "recall, F1-score, ROC-AUC, and sensitivity should also "
        "be considered."
    )
# -------------------------
# ABOUT
# -------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About Brain Stroke AI Agent")

    st.subheader("AI Agent for Brain Stroke Prediction")

    st.write("""
    ### 🧠 Project Overview

    This project is an AI-based Brain Stroke Prediction system
    using Electronic Health Records (EHR) and Clinical Decision Support.

    The system uses multiple Machine Learning and Deep Learning
    models to predict stroke risk.

    ### 🤖 Models Used

    ✅ Deep Neural Network (DNN)

    ✅ TabNet

    ✅ FT-Transformer

    ### 📊 Dataset

    Stroke Prediction Dataset from Kaggle.

    ### 🩺 Clinical Decision Support

    The application integrates Gemini AI to provide
    educational clinical decision-support information
    based on patient details.

    ### 🛠️ Technologies Used

    - Python
    - Streamlit
    - TensorFlow
    - PyTorch
    - TabNet
    - Gemini AI
    - Pandas
    - NumPy
    - Scikit-learn

    ### 👩‍💻 Developer

    **Muskan Shaik**

    B.Tech CSD

    ### ⚠️ Disclaimer

    This application is intended for educational purposes
    and is not a replacement for professional medical advice.
    """)

    # -------------------------
    # FOOTER
    # -------------------------

    st.markdown(
        """
        ---
        🧠 **Brain Stroke AI Agent**  
        AI-powered stroke risk prediction & clinical decision support

        Developed by **Muskan Shaik • B.Tech CSD**
        """
    )