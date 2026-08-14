# 🧠 Brain Stroke AI Agent

> **AI-Powered Stroke Risk Prediction & Clinical Decision Support System**  
> Built with Streamlit, PyTorch, TensorFlow, TabNet, FT-Transformer, and Google Gemini AI.

---

## 🌟 Overview

The **Brain Stroke AI Agent** is an end-to-end medical decision support and risk estimation platform. Using Electronic Health Records (EHR), the application leverages three state-of-the-art Deep Learning models to analyze patient health factors (such as age, glucose levels, BMI, hypertension, heart disease, and smoking history) and generate accurate stroke risk probabilities.

In addition, the application incorporates **Google Gemini AI** to provide educational clinical recommendations and interactive health chat support.

---

## 🔥 Key Features

- **📊 Stroke Risk Prediction**: Estimate stroke risk using 3 customizable deep learning architectures:
  - **DNN (Deep Neural Network)** - High accuracy feed-forward network.
  - **TabNet** - Attention-based neural network architecture optimized for tabular healthcare data.
  - **FT-Transformer (Feature Tokenizer Transformer)** - Transformer-based architecture adapted for numerical and categorical medical features.
- **🤖 Clinical Decision Support**: Generates personalized, educational clinical summaries, lifestyle recommendations, and emergency warning signs powered by Gemini AI.
- **💬 AI Health Assistant**: Conversational health chatbot to answer questions on stroke symptoms, risk factors, prevention, and medical terminology.
- **📈 Model Comparison Dashboard**: Comparative performance visualization, accuracy benchmarks, and evaluation metrics across DNN, TabNet, and FT-Transformer.
- **ℹ️ Educational Disclaimer**: Built with clinical safety guidelines to clarify educational/decision-support intent.

---

## 🏗️ Project Architecture

```
BrainStrokeAI_CollegeProject/
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── .env                       # Local environment variables (API keys)
├── .streamlit/
│   └── config.toml            # Streamlit theme & server configuration
├── models/
│   ├── dnn_model.keras        # Trained Keras DNN model
│   ├── tabnet_model.zip       # Trained TabNet Classifier model
│   ├── ft_transformer.pth     # Trained PyTorch FT-Transformer model weights
│   ├── scaler.pkl             # Scikit-learn StandardScaler instance
│   └── encoders.pkl           # Categorical LabelEncoders dictionary
├── src/
│   ├── preprocess.py          # Data preprocessing script
│   ├── train_dnn.py           # DNN training code
│   ├── train_tabnet.py        # TabNet model training code
│   └── train_fttransformer.py # FT-Transformer model training code
└── data/
    └── healthcare-dataset-stroke-data.csv # EHR dataset
```

---

## 💻 Local Setup & Execution

### 1. Prerequisites
- Python 3.10+ installed on your computer.
- A Google Gemini API Key (Get one from [Google AI Studio](https://aistudio.google.com/)).

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muskanshaik-07/BrainStrokeAI-CollegeProject.git
   cd BrainStrokeAI-CollegeProject
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Cloud

You can deploy this application live on **Streamlit Community Cloud** in 4 simple steps:

1. **Go to Streamlit Cloud:**
   Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

2. **Deploy a New App:**
   - Click **New app**.
   - Select Repository: `muskanshaik-07/BrainStrokeAI-CollegeProject`
   - Select Branch: `master` (or `main`)
   - Main file path: `app.py`

3. **Add Gemini API Key in Secrets:**
   - Before clicking Deploy, click **Advanced settings...** (or go to App Settings → **Secrets**).
   - Paste the following:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key_here"
     ```

4. **Launch Application:**
   Click **Deploy!**. Streamlit will install dependencies from `requirements.txt` and launch your live public web app.

---

## 🏆 Model Performance Summary

| Model Architecture | Test Accuracy |
| :--- | :---: |
| **Deep Neural Network (DNN)** | **93.93%** |
| **TabNet Classifier** | **93.84%** |
| **FT-Transformer** | **93.93%** |

---

## 👩‍💻 Developer Info

- **Developer**: Muskan Shaik
- **Degree**: B.Tech Computer Science & Data Science (CSD)
- **Repository**: [github.com/muskanshaik-07/BrainStrokeAI-CollegeProject](https://github.com/muskanshaik-07/BrainStrokeAI-CollegeProject)

---

## ⚠️ Medical Disclaimer

This software is designed solely for educational, research, and decision-support demonstration purposes. It does not provide medical diagnoses, treatment plans, or prescriptions. Always consult a qualified healthcare provider for clinical medical advice.
