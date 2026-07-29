import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #dcdcdc;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }

    .stButton>button {
        width: 100%;
        background-color: #00c6ff;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        height: 3em;
        border: none;
    }

    .stButton>button:hover {
        background-color: #0072ff;
        color: white;
    }

    /* Labels White */
    label {
        color: white !important;
        font-weight: bold !important;
    }

    /* Input Text White */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stSlider {
        color: white !important;
    }

    /* Dropdown Text */
    div[data-baseweb="select"] > div {
    color: white !important;
    background-color: rgba(255,255,255,0.1) !important;
    }

    /* Number Input Box */
    .stNumberInput div div input {
    background-color: white !important;
    color: black !important;
    }
    
    input {
    color: black !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


model = joblib.load("churn_model.pkl")


st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether a customer is likely to churn based on customer behavior</div>',
    unsafe_allow_html=True
)


with st.container():

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    weekly_active_days = st.slider(
        "Weekly Active Days",
        0,
        7,
        5
    )

    last_login_days_ago = st.number_input(
        "Last Login Days Ago",
        min_value=0,
        value=2
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Auto Pay", "Manual Payment"]
    )


gender_value = 1 if gender == "Male" else 0
payment_value = 0 if payment_method == "Auto Pay" else 1


if st.button("Predict Churn"):

    input_data = pd.DataFrame([[
        gender_value,
        age,
        weekly_active_days,
        last_login_days_ago,
        payment_value
    ]], columns=[
        "gender",
        "age",
        "weekly_active_days",
        "last_login_days_ago",
        "payment_method"
    ])

    
    prediction_proba = model.predict_proba(input_data)[0][1]

    
    risk_score = 0

    if weekly_active_days <= 1:
        risk_score += 1

    if last_login_days_ago >= 60:
        risk_score += 1

    if payment_method == "Manual Payment":
        risk_score += 1

    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

    
    if prediction_proba > 0.4 or risk_score >= 2:

        st.error("⚠️ Customer is likely to churn")

        st.progress(min(int(prediction_proba * 100), 100))

        st.write(f"### Churn Probability: {prediction_proba:.2%}")

        st.write("#### Suggested Action:")
        st.write("- Offer discounts")
        st.write("- Improve engagement")
        st.write("- Contact customer support team")

    else:

        st.success("✅ Customer is not likely to churn")

        st.progress(min(int((1 - prediction_proba) * 100), 100))

        st.write(f"### Customer Retention Probability: {(1 - prediction_proba):.2%}")

        st.write("#### Positive Indicators:")
        st.write("- Active customer")
        st.write("- Good engagement")
        st.write("- Lower churn risk")

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    "<center>Built using Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)
