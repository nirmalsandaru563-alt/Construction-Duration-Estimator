import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm

st.set_page_config(page_title="Tender Feasibility App", layout="centered")

st.title("🚜 Contractual Deadline Feasibility Check")
st.markdown("### Risk-Based Tender Stage Evaluation Tool")
st.write("---")

@st.cache_resource
def load_framework():
    return joblib.load('construction_model_bundle.pkl')

bundle = load_framework()
model_pipeline = bundle['pipeline']
rmse = bundle['rmse']
categorical_options = bundle['categorical_options']

st.markdown("#### 🏢 Step 1: Input Project Tender Configurations")
col1, col2 = st.columns(2)

with col1:
    project_type = st.selectbox("Project Category", categorical_options['project_type'])
    client_type = st.selectbox("Client Profile", categorical_options['client_type'])
    procurement_method = st.selectbox("Procurement Route", categorical_options['procurement_method'])
    contractor_tier = st.selectbox("Target Contractor Status", categorical_options['contractor_tier'])
    soil_condition = st.selectbox("Site Soil Condition", categorical_options['soil_condition'])
    complexity_level = st.selectbox("Design Complexity Level", categorical_options['complexity_level'])

with col2:
    year_started = st.number_input("Proposed Commencement Year", min_value=2020, max_value=2035, value=2026)
    contract_value = st.number_input("Estimated Contract Value (LKR Millions)", min_value=1.0, max_value=5000.0, value=450.0)
    gfa = st.number_input("Gross Floor Area (m²)", min_value=10, max_value=50000, value=1500)
    num_storeys = st.slider("Total Number of Storeys", min_value=1, max_value=60, value=6)
    num_basements = st.slider("Number of Basement Levels", min_value=0, max_value=5, value=0)
    num_subcontractors = st.slider("Estimated Specialist Subcontractors", min_value=0, max_value=60, value=12)

st.write("---")
st.markdown("#### 🎯 Step 2: Set Proposed Contractual Boundary")
target_days = st.number_input("Enter Client's Desired Contract Duration (Days)", min_value=50, max_value=2000, value=600)

if st.button("📊 Run Target Feasibility Assessment", type="primary"):
    input_dict = {
        'year_started': [year_started],
        'project_type': [project_type],
        'client_type': [client_type],
        'procurement_method': [procurement_method],
        'contractor_tier': [contractor_tier],
        'contract_value_LKR_M': [contract_value],
        'gross_floor_area_m2': [gfa],
        'num_storeys': [num_storeys],
        'num_basements': [num_basements],
        'num_subcontractors': [num_subcontractors],
        'soil_condition': [soil_condition],
        'complexity_level': [complexity_level]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    predicted_mean = model_pipeline.predict(input_df)[0]
    
    probability_of_completion = norm.cdf(target_days, loc=predicted_mean, scale=rmse)
    prob_percentage = int(probability_of_completion * 100)
    
    st.write("---")
    st.markdown("### 📋 Evaluation Decision Framework")
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Ensemble Expected Target Base", f"{int(predicted_mean)} Days")
    col_stat2.metric("Probability of Meeting Target", f"{prob_percentage}%")
    
    st.progress(probability_of_completion)
    
    if prob_percentage >= 80:
        st.success(f"### 👍 Highly Feasible Frame ({prob_percentage}% Confidence Level)\n"
                   f"The proposed schedule of **{target_days} days** is mathematically viable. "
                   f"Historical baseline performance suggests ample buffer space to manage standard site challenges.")
    elif prob_percentage >= 50:
        st.warning(f"### ⚠️ Moderately Feasible Frame ({prob_percentage}% Confidence Level)\n"
                   f"The target timeline of **{target_days} days** sits close to historical performance parameters. "
                   f"While viable, typical variation or standard delivery overruns could shift the timeline. High monitoring diligence advised.")
    else:
        st.error(f"### 🚨 Aggressive Schedule Risk ({prob_percentage}% Confidence Level)\n"
                 f"The contractual deadline of **{target_days} days** presents an elevated delivery risk. "
                 f"Our ensemble framework determines that achieving this target has a low likelihood of success. "
                 f"It is highly recommended to extend the contract target closer to **{int(predicted_mean)} days** or renegotiate procurement variables.")
