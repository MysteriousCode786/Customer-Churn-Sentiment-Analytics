import pickle 
import nltk 
import pandas as pd 
import plotly.express as px
import streamlit as st 
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER dictionary package (safe check for local environment)
nltk.download("vader_lexicon", quiet = True)

# Set up the Page Configuartion
st.set_page_config(
    page_title = "Customer Churn & Sentiment Analytics", layout = "wide"
)

st.title("📊 Customer Churn & Sentiment Analytics Dashboard")
st.markdown(
    "Predict whether a customer will leave the platform based on behavioral metrics and review text."
)
st.write("---")

# Load the pre-trained Random Forest Model
try:
    with open("churn_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error(
        "Error: 'churn_model.pkl not found in this folder."
    )
    st.stop()

# Initializa VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Sidebar Panel for Input Controls
st.sidebar.header("🔍 Input Customer Details")
st.sidebar.markdown("Adjust customer attributes below:")

# All numeric parameters 
age = st.sidebar.slider("Customer Age", min_value=18, max_value=65, value=35)
tenure = st.sidebar.slider(
    "Tenure (In Months)", min_value=1, max_value=24, value=12
)
logins = st.sidebar.slider(
    "Monthly Logins Frequency", min_value=1, max_value=30, value=15
)
tickets = st.sidebar.slider(
    "Support Tickets Raised", min_value=0, max_value=4, value=1
)

# Main Page Display Area
# Divide main viewport into two wide layout columns
col1, col2 = st.columns([1, 1], gap = "large")

with col1:
    st.header("📝 Customer Review Text")
    # Unstructured text input for sentiment analysis
    review_text = st.text_area(
        "Customer Review Text Input",
        value = "Type or paste user feedback here...",
        height = 120,
    )

    # Process sentiment instantly on user keystrokes
    sentiment_score = sia.polarity_scores(review_text)["compound"]
    st.metric(
        label="Extracted Sentiment Analysis Score", value=f"{sentiment_score:.4f}"
    )

with col2:
    st.header("🎯 Prediction & Analysis Output")

    # Structure the input variables into a DataFrame matching model columns
    input_features = pd.DataFrame(
        [[age, tenure, logins, tickets, sentiment_score]],
        columns = [
            "Age",
            "TenureMonths",
            "MonthlyLogins",
            "SupportTickets",
            "SentimentScore",
        ],
    )

    # Predict Churn using the loaded model
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]

    st.write("### Inference Status Result:")
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk Status: Likely to Churn!** (Confidence: {prediction_proba[1] * 100:.1f}%)"
        )
    else:
        st.success(
            f"✅ **Safe Status: Likely to stay.** (Confidence: {prediction_proba[0] * 100:.1f}%)"
        )

st.write("---")
st.subheader("📈 Customer Health Metrics Visualization")

# Add a visual indicator graph using Plotly
metrics_df = pd.DataFrame(
    {
        "Metric": ["Monthly Logins", "Support Tickets", "Sentiment Score (Scaled)"],
        "Value": [logins, tickets, sentiment_score * 10],
    }
)
fig = px.bar(
    metrics_df,
    x = "Metric",
    y = "Value",
    color = "Metric",
    title = "Comparative Overview of Customer Key Indicators",
    template = "plotly_white"
)

fig.update_layout(
    yaxis_title="Metric Value / Intensity Scale (-10 to +10)",
    xaxis_title="" 
)

st.plotly_chart(fig, use_container_width = True)