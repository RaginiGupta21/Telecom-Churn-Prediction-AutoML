import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# loading the files saved from the notebook
@st.cache_resource
def load_files():
    model = joblib.load("models/churn_model.pkl")
    columns = joblib.load("models/model_columns.pkl")
    threshold = joblib.load("models/threshold.pkl")
    return model, columns, threshold

model, model_columns, threshold = load_files()

# same feature code as the notebook
addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
              'TechSupport', 'StreamingTV', 'StreamingMovies']

def add_new_features(data):
    data['num_addons'] = (data[addon_cols] == 'Yes').sum(axis=1)
    data['auto_pay'] = data['PaymentMethod'].str.contains('automatic').astype(int)
    data['fiber_no_support'] = ((data['InternetService'] == 'Fiber optic') & (data['TechSupport'] != 'Yes')).astype(int)
    return data


st.title("Telecom Customer Churn Predictor")
st.write("Enter a customer's details to see how likely they are to leave, why, and what the company could do.")
st.caption("Model: LightGBM found by FLAML AutoML, trained on the IBM Telco dataset. This is a student project and the offer costs used for the threshold are assumptions.")

st.subheader("Customer details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account**")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", min_value=18.0, max_value=120.0, value=70.0, step=0.5)

with col2:
    st.markdown("**Personal and phone**")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has a partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "No":
        multiple = "No phone service"
    else:
        multiple = st.selectbox("Multiple lines", ["No", "Yes"])

with col3:
    st.markdown("**Internet services**")
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    if internet == "No":
        st.info("No internet, so the internet services below are set to 'No internet service'.")
        security = backup = protection = tech = tv = movies = "No internet service"
    else:
        security = st.selectbox("Online security", ["No", "Yes"])
        backup = st.selectbox("Online backup", ["No", "Yes"])
        protection = st.selectbox("Device protection", ["No", "Yes"])
        tech = st.selectbox("Tech support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming movies", ["No", "Yes"])

if st.button("Predict churn risk", type="primary"):

    # total charges is not asked, so it is estimated as monthly charges x tenure
    total = monthly * tenure

    customer = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
        'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly, 'TotalCharges': total}])

    # same steps as training
    customer = add_new_features(customer)
    customer = pd.get_dummies(customer, dtype=int)
    customer = customer.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(customer)[0, 1]

    if prob >= threshold:
        risk = "HIGH"
    elif prob >= threshold / 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.divider()
    st.subheader("Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Churn probability", f"{prob * 100:.1f} %")
    r2.metric("Risk level", risk)
    r3.metric("Send retention offer?", "YES" if prob >= threshold else "NO")

    if risk == "HIGH":
        st.error(f"This customer is above the threshold ({threshold}), so they should get a retention offer.")
    elif risk == "MEDIUM":
        st.warning("This customer is not above the threshold but should be watched.")
    else:
        st.success("This customer is unlikely to leave.")

    # simple rules for what the company could do
    if risk != "LOW":
        st.subheader("Suggested actions")
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount to move to a one-year or two-year contract.")
        if internet != "No" and security == "No":
            tips.append("Offer a free trial of online security.")
        if internet != "No" and tech == "No":
            tips.append("Offer free tech support for a few months.")
        if payment == "Electronic check":
            tips.append("Encourage automatic payment (bank transfer or credit card).")
        if tenure < 12:
            tips.append("The customer is new, so make a welcome call and ask how they are finding the service.")
        if len(tips) == 0:
            tips.append("No specific action found, keep monitoring this customer.")
        for tip in tips:
            st.write("- " + tip)

    # shap explanation for this customer
    st.subheader("Why does the model say this?")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(customer)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        contrib = pd.Series(shap_values[0], index=customer.columns)
        top = contrib[contrib.abs().sort_values(ascending=False).head(8).index].sort_values()

        fig, ax = plt.subplots(figsize=(8, 4))
        top.plot(kind='barh', color=['#C42B1E' if v > 0 else '#3A8524' for v in top], ax=ax)
        ax.set_xlabel("shap value (red = pushes towards churn, green = pushes towards staying)")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.write("Could not draw the explanation chart:", e)

with st.expander("How does this work?"):
    st.write("""
    - The model gives a churn probability for the customer.
    - The threshold (""" + str(threshold) + """) was chosen in the notebook by testing which value gives the highest profit for a retention campaign. It is lower than 0.5 because an offer is cheap compared to losing a customer.
    - The chart uses SHAP values, which show how much each feature pushed this prediction up or down.
    - The suggested actions are simple rules, they are not predicted by the model.
    """)