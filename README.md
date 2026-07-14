# 📊 Multi-Modal Customer Churn Prediction & Sentiment Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-VADER-green?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)

An end-to-end Machine Learning operational system and intelligent monitoring dashboard designed to identify retention risks in subscription-based customer portfolios. 

Unlike traditional isolated tabular analysis, this architecture implements a **Multi-Modal Framework** that evaluates structured behavioral attributes (usage logins, customer service friction metrics) alongside unstructured real-time textual feedback simultaneously.

🔗 **Live Deployment Link:** [👉 Click Here to Test the Live Interactive Application Dashboard] https://customer-churn-sentiment-analytics-6gv9q7bgcuwyxhccgomico.streamlit.app/

---

## 🎯 Strategic Core Business Capabilities
* **Dual-Engine Feature Pipeline**: Merges native customer engagement usage records with raw text parameters processed on-the-fly via natural language parsing models.
* **Real-Time Sentiment Quantification**: Leverages a rule-based lexical analyzer (`nltk.sentiment.vader`) to convert qualitative sentiment inputs into high-precision continuous numerical scores.
* **Deterministic Risk Allocation**: Leverages a serialized production-ready Random Forest Classifer model optimized to emit continuous feature classification probabilities (`predict_proba()`).
* **Sleek Reactive User Experience**: Features an enterprise-standard structural side-panel design layout separating data control loops from core analytical workspace graphics.

---

## 🏗️ System Pipeline Architecture

The workflow shifts multi-layered properties smoothly across distinct algorithmic computational boundaries:
[Raw User Properties] ----> [Sidebar Controls (Streamlit Engine)]
│
[Unstructured Feedback] --> [NLTK VADER Extraction Module] ────> [Vector Matrix Frame (Pandas Dataframe)]
│
▼
[Interactive Plotly Visual Analytics] <─── [System Output Banners] <─── [Inference Model (sklearn Array Execution)]

---

## 📂 Production Code Directory Matrix
├── app.py                            # Reactive application UI framework & core inference runtime script
├── churn_model.pkl                   # Frozen, binary serialized Random Forest classifier model state
├── requirements.txt                  # Consolidated library deployment target manifest
├── README.md                         # Detailed project roadmap and architectural design overview
├── customer_churn_data.csv           # Initial simulated operational engagement parameters dataset
└── customer_churn_with_sentiment.csv  # Preprocessed model dataset housing mapped textual compound weights

---

## 🔬 Boundary Stress Testing & Conflict Analysis
To ensure model robustness, the system architecture was subjected to **Feature Correlation Conflict Scenarios** to validate prediction boundaries:

* **High-Engagement Friction (Case A)**: Inputting strong structural attributes (High logins, zero service tickets) combined with deeply negative text reviews. The model properly prioritized extreme qualitative frustration weightages (`Sentiment Score < -0.8`), successfully firing the high-risk alert banner state: `⚠️ High Risk Status: Likely to Churn!`.
* **Low-Engagement Affinity (Case B)**: Inputting elevated friction parameters (High ticket logs, low login frequencies) paired with highly appreciative feedback strings. The backend pipeline safely recalibrated target probabilities toward heavy textual affinity flags, stabilizing localized user behavior outputs.

---

## ⚙️ Local Deployment & Environment Setup

Execute these instructions in your local native system console to establish an identical workspace runtime instance:

## 🏃‍♂️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MysteriousCode786/Customer-Churn-Sentiment-Analytics.git
   cd Customer-Churn-Sentiment-Analytics
2. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Launch the Streamlit Dashboard:**
   ```bash
   streamlit run app.py