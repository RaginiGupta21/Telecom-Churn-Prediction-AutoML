import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📞", layout="wide")

# ---------------- style (css) ----------------
st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #0F2A52, #1F5FBF); padding: 28px 34px; border-radius: 16px; margin-bottom: 16px;}
.hero h1 {color: white !important; margin: 0; padding: 0; font-size: 2.1rem;}
.hero p {color: #DCE8FA; margin: 8px 0 0 0; font-size: 1.05rem;}
.stat {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 14px 10px; text-align: center;}
.stat .num {font-size: 1.6rem; font-weight: 700; color: #1F5FBF;}
.stat .label {font-size: 0.85rem; opacity: 0.8;}
.risk-card {border-radius: 16px; padding: 22px 16px; text-align: center; color: white;}
.risk-card .tag {font-size: 1.05rem; font-weight: 600; letter-spacing: 1px;}
.risk-card .pct {font-size: 3.3rem; font-weight: 800; line-height: 1.15;}
.high {background: linear-gradient(135deg, #C42B1E, #E4572E);}
.medium {background: linear-gradient(135deg, #D98E04, #F2B134);}
.low {background: linear-gradient(135deg, #2E7D32, #4CAF50);}
.action {background: rgba(31,95,191,0.10); border-left: 4px solid #1F5FBF; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------- loading the saved files ----------------
@st.cache_resource
def load_files():
    model = joblib.load("models/churn_model.pkl")
    columns = joblib.load("models/model_columns.pkl")
    threshold = joblib.load("models/threshold.pkl")
    return model, columns, threshold

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model, model_columns, threshold = load_files()


# ---------------- same feature code as the notebook ----------------
addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
              'TechSupport', 'StreamingTV', 'StreamingMovies']

def add_new_features(data):
    data['num_addons'] = (data[addon_cols] == 'Yes').sum(axis=1)
    data['auto_pay'] = data['PaymentMethod'].str.contains('automatic').astype(int)
    data['fiber_no_support'] = ((data['InternetService'] == 'Fiber optic') & (data['TechSupport'] != 'Yes')).astype(int)
    return data


# ---------------- small helper functions ----------------
def nice_name(col):
    # Contract_Month-to-month -> Contract: Month-to-month
    if col in ['num_addons', 'auto_pay', 'fiber_no_support']:
        return col.replace('_', ' ')
    return col.replace('_', ': ', 1)

def risk_meter(prob, threshold):
    p = min(max(prob * 100, 1), 99)
    t = threshold * 100
    return f"""
    <div style="position:relative; margin:34px 6px 8px 6px;">
        <div style="height:16px; border-radius:8px; background:linear-gradient(90deg,#3A8524,#F2C94C,#C42B1E);"></div>
        <div style="position:absolute; left:{t}%; top:-8px; height:32px; border-left:2px dashed #123463;"></div>
        <div style="position:absolute; left:{t}%; top:-30px; transform:translateX(-50%); font-size:12px; opacity:0.8; white-space:nowrap;">offer threshold {t:.0f}%</div>
        <div style="position:absolute; left:{p}%; top:-5px; width:26px; height:26px; margin-left:-13px; border-radius:50%; background:white; border:5px solid #123463;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:12px; opacity:0.7; margin:0 6px;"><span>0%</span><span>100%</span></div>
    """


# ---------------- default values and examples for the form ----------------
defaults = {'tenure': 12, 'contract': 'Month-to-month', 'paperless': 'Yes', 'payment': 'Electronic check',
            'monthly': 70.0, 'gender': 'Female', 'senior': 'No', 'partner': 'No', 'dependents': 'No',
            'phone': 'Yes', 'multiple': 'No', 'internet': 'Fiber optic', 'security': 'No', 'backup': 'No',
            'protection': 'No', 'tech': 'No', 'tv': 'No', 'movies': 'No'}

risky_example = {'tenure': 2, 'contract': 'Month-to-month', 'paperless': 'Yes', 'payment': 'Electronic check',
                 'monthly': 85.0, 'gender': 'Female', 'senior': 'No', 'partner': 'No', 'dependents': 'No',
                 'phone': 'Yes', 'multiple': 'No', 'internet': 'Fiber optic', 'security': 'No', 'backup': 'No',
                 'protection': 'No', 'tech': 'No', 'tv': 'No', 'movies': 'No'}

loyal_example = {'tenure': 60, 'contract': 'Two year', 'paperless': 'No', 'payment': 'Bank transfer (automatic)',
                 'monthly': 60.0, 'gender': 'Male', 'senior': 'No', 'partner': 'Yes', 'dependents': 'Yes',
                 'phone': 'Yes', 'multiple': 'Yes', 'internet': 'DSL', 'security': 'Yes', 'backup': 'Yes',
                 'protection': 'Yes', 'tech': 'Yes', 'tv': 'No', 'movies': 'No'}

for key in defaults:
    if key not in st.session_state:
        st.session_state[key] = defaults[key]

def load_example(values):
    for key in values:
        st.session_state[key] = values[key]


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write("A student project that predicts which telecom customers may leave, explains why, and suggests what to offer them.")
    st.markdown("**How to use**")
    st.write("1. Fill in the customer details, or load an example.\n2. The result updates by itself.\n3. Read the reasons and suggested actions.")
    st.markdown("**Model**")
    st.write("LightGBM found with FLAML AutoML, trained on the IBM Telco dataset (7,043 customers).")
    st.markdown(f"**Offer threshold:** {threshold}")
    st.markdown("[GitHub repository](https://github.com/RaginiGupta21/Telecom-Churn-Prediction-AutoML)")
    st.caption("The offer costs behind the threshold are assumptions, not real company data.")


# ---------------- header ----------------
st.markdown("""
<div class="hero">
    <h1>Telecom Customer Churn Predictor</h1>
    <p>Find out which customers are about to leave, why they might leave, and what to offer them.</p>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
s1.markdown('<div class="stat"><div class="num">0.847</div><div class="label">ROC-AUC on test data</div></div>', unsafe_allow_html=True)
s2.markdown('<div class="stat"><div class="num">84%</div><div class="label">of churners caught at the chosen threshold</div></div>', unsafe_allow_html=True)
s3.markdown('<div class="stat"><div class="num">7,043</div><div class="label">customers in the dataset</div></div>', unsafe_allow_html=True)
s4.markdown('<div class="stat"><div class="num">AutoML</div><div class="label">model chosen by FLAML</div></div>', unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3 = st.tabs(["Predict", "How it works", "Model results"])


# =============== TAB 1: predict ===============
with tab1:
    st.write("Try an example, or enter your own customer:")
    b1, b2, b3, b4 = st.columns([1.3, 1.3, 0.8, 3])
    b1.button("Example: likely to churn", on_click=load_example, args=(risky_example,))
    b2.button("Example: loyal customer", on_click=load_example, args=(loyal_example,))
    b3.button("Reset", on_click=load_example, args=(defaults,))

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("**Account and billing**")
            tenure = st.slider("Tenure (months)", 0, 72, key="tenure")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
            paperless = st.selectbox("Paperless billing", ["Yes", "No"], key="paperless")
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)", "Credit card (automatic)"], key="payment")
            monthly = st.number_input("Monthly charges", min_value=18.0, max_value=120.0, step=0.5, key="monthly")

    with col2:
        with st.container(border=True):
            st.markdown("**Personal and phone**")
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
            senior = st.selectbox("Senior citizen", ["No", "Yes"], key="senior")
            partner = st.selectbox("Has a partner", ["No", "Yes"], key="partner")
            dependents = st.selectbox("Has dependents", ["No", "Yes"], key="dependents")
            phone = st.selectbox("Phone service", ["Yes", "No"], key="phone")
            if phone == "No":
                multiple = "No phone service"
            else:
                multiple = st.selectbox("Multiple lines", ["No", "Yes"], key="multiple")

    with col3:
        with st.container(border=True):
            st.markdown("**Internet services**")
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"], key="internet")
            if internet == "No":
                st.info("No internet, so the services below are set to 'No internet service'.")
                security = backup = protection = tech = tv = movies = "No internet service"
            else:
                security = st.selectbox("Online security", ["No", "Yes"], key="security")
                backup = st.selectbox("Online backup", ["No", "Yes"], key="backup")
                protection = st.selectbox("Device protection", ["No", "Yes"], key="protection")
                tech = st.selectbox("Tech support", ["No", "Yes"], key="tech")
                tv = st.selectbox("Streaming TV", ["No", "Yes"], key="tv")
                movies = st.selectbox("Streaming movies", ["No", "Yes"], key="movies")

    # ----- prediction (updates every time an input changes) -----
    total = monthly * tenure    # total charges is not asked, so it is estimated

    customer = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
        'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly, 'TotalCharges': total}])

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

    # ----- results -----
    st.write("")
    st.subheader("Result")
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown(f"""
        <div class="risk-card {risk.lower()}">
            <div class="tag">{risk} RISK</div>
            <div class="pct">{prob * 100:.1f}%</div>
            <div>chance this customer leaves</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(risk_meter(prob, threshold), unsafe_allow_html=True)

        if risk == "HIGH":
            st.error("Above the threshold, so this customer should get a retention offer.")
        elif risk == "MEDIUM":
            st.warning("Below the threshold, but worth keeping an eye on.")
        else:
            st.success("Unlikely to leave. No offer needed.")

    with right:
        st.markdown("**Suggested actions**")
        if risk == "LOW":
            st.markdown('<div class="action">No action needed, keep the customer happy with the current service.</div>', unsafe_allow_html=True)
        else:
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
                st.markdown(f'<div class="action">{tip}</div>', unsafe_allow_html=True)
        st.caption("These actions are simple rules, they are not predicted by the model.")

    # ----- shap explanation -----
    st.subheader("Why does the model say this?")
    try:
        explainer = load_explainer(model)
        shap_values = explainer.shap_values(customer)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        contrib = pd.Series(shap_values[0], index=customer.columns)

        up = contrib[contrib > 0].sort_values(ascending=False).head(3)
        down = contrib[contrib < 0].sort_values().head(2)
        if len(up) > 0:
            st.write("**Pushing towards leaving:** " + ", ".join([nice_name(c) for c in up.index]))
        if len(down) > 0:
            st.write("**Pushing towards staying:** " + ", ".join([nice_name(c) for c in down.index]))

        top = contrib[contrib.abs().sort_values(ascending=False).head(8).index].sort_values()
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh([nice_name(c) for c in top.index], top.values,
                color=['#C42B1E' if v > 0 else '#3A8524' for v in top])
        ax.axvline(0, color='grey', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("effect on churn risk  (red = towards leaving, green = towards staying)")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.write("Could not draw the explanation chart:", e)


# =============== TAB 2: how it works ===============
with tab2:
    st.subheader("How does this app work?")
    st.markdown(f"""
    1. **You enter a customer's details.** The app turns them into the same features the model was trained on.
    2. **The model gives a churn probability.** It is a LightGBM model that FLAML AutoML picked after trying LightGBM, XGBoost, Random Forest and Extra Trees.
    3. **The probability is compared with a threshold ({threshold}).** The threshold is lower than the usual 0.5 because I tested which value gives the highest profit for a retention campaign, using assumed costs. An offer is cheap compared to losing a customer.
    4. **SHAP explains the prediction.** It shows how much each feature pushed this customer's risk up or down.
    5. **Suggested actions** are simple rules based on the customer's details.
    """)
    st.info("This is a student project. The costs behind the threshold are assumptions, and the model only shows what it learned from one public dataset.")
    st.markdown("**Built with:** Python, pandas, scikit-learn, FLAML, LightGBM, SHAP, Streamlit")


# =============== TAB 3: model results ===============
with tab3:
    st.subheader("How good is the model?")
    st.markdown("""
    | Model | ROC-AUC | Recall | Accuracy |
    |---|---|---|---|
    | FLAML (LightGBM) | 0.847 | 53% | 80.0% |
    | XGBoost | 0.843 | 52% | 80.5% |
    | Logistic Regression | 0.841 | 56% | 80.4% |
    | LightGBM (balanced weights) | 0.839 | 78% | 76.4% |
    | Random Forest | 0.837 | 52% | 80.0% |
    | LightGBM (hand-set) | 0.836 | 52% | 80.0% |
    """)
    st.caption("Scores on 1,409 test customers at the default 0.5 threshold. Lowering the threshold to 0.2 raises recall to about 84%.")

    def show_image(path, caption):
        if os.path.exists(path):
            st.image(path, caption=caption)

    r1, r2 = st.columns(2)
    with r1:
        show_image("images/threshold_vs_profit.png", "Why a lower threshold gives more profit (simulated costs)")
        show_image("images/shap_summary.png", "What drives churn (SHAP)")
    with r2:
        show_image("images/confusion_matrix.png", "Confusion matrix at the chosen threshold")
        show_image("images/model_comparison.png", "Model comparison with 95% intervals")