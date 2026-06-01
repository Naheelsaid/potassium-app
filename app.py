import streamlit as st
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(
    page_title="IV Potassium Calculator",
    page_icon="💊",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { padding: 0.5rem; }
    .block-container { padding: 1rem; max-width: 100%; }
    .stButton>button {
        width: 100%;
        height: 3.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 12px;
        background-color: #2C3E7A;
        color: white;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover { background-color: #1a2a5e; }
    .metric-card {
        background: linear-gradient(135deg, #2C3E7A, #1a2a5e);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        margin: 0.4rem 0;
        box-shadow: 0 4px 15px rgba(44,62,122,0.3);
    }
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.85;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-unit  { font-size: 0.85rem; opacity: 0.75; }
    .section-header {
        background: #f0f4ff;
        border-left: 4px solid #2C3E7A;
        padding: 0.5rem 0.8rem;
        border-radius: 0 8px 8px 0;
        font-weight: bold;
        color: #2C3E7A;
        margin: 1rem 0 0.5rem 0;
        font-size: 1rem;
    }
    .alert-success {
        background: #d4edda; border: 1px solid #28a745;
        border-radius: 10px; padding: 1rem; color: #155724;
        text-align: center; font-weight: bold;
        font-size: 1rem; margin-top: 0.5rem;
    }
    .alert-warning {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 10px; padding: 1rem; color: #856404;
        text-align: center; font-weight: bold;
        font-size: 1rem; margin-top: 0.5rem;
    }
    .alert-danger {
        background: #f8d7da; border: 1px solid #dc3545;
        border-radius: 10px; padding: 1rem; color: #721c24;
        text-align: center; font-weight: bold;
        font-size: 1rem; margin-top: 0.5rem;
    }
    div[data-testid="stNumberInput"] input {
        font-size: 1.05rem; height: 2.8rem; border-radius: 8px;
    }
    div[data-testid="stSelectbox"] select {
        font-size: 1.05rem; height: 2.8rem; border-radius: 8px;
    }
    .stCheckbox label { font-size: 1rem; }
    h1 { font-size: 1.5rem !important; text-align: center; color: #2C3E7A; }
    h3 { font-size: 1.1rem !important; color: #2C3E7A; }
    </style>
""", unsafe_allow_html=True)

# ── Google Sheets connection ───────────────────────────────────────────────
@st.cache_resource
def get_gsheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    # Load credentials from Streamlit secrets
    creds_dict = st.secrets["gcp_service_account"]
    creds      = ServiceAccountCredentials.from_json_keyfile_dict(
                    dict(creds_dict), scope)
    client     = gspread.authorize(creds)
    sheet      = client.open('potassium_patients').sheet1
    return sheet

def save_to_gsheet(row):
    try:
        sheet = get_gsheet()
        # Add header if sheet is empty
        if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
            sheet.append_row(list(row.keys()))
        sheet.append_row(list(row.values()))
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

def load_from_gsheet():
    try:
        sheet   = get_gsheet()
        records = sheet.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Load error: {e}")
        return pd.DataFrame()

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_potassium_model.cbm')
    return model

model = load_model()

# ── Navigation ─────────────────────────────────────────────────────────────
page = st.sidebar.selectbox("📄 Navigate",
                             ["💊 Calculator", "📊 External Validation"])

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Calculator
# ══════════════════════════════════════════════════════════════════════════
if page == "💊 Calculator":

    st.markdown("# 💊 IV Potassium Response Calculator")
    st.markdown(
        "<p style='text-align:center; color:#666; font-size:0.9rem;'>"
        "Predicts serum K⁺ increment per 10 mEq IV KCl</p>",
        unsafe_allow_html=True)

    # ── Demographics ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>👤 Demographics</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age     = st.number_input("Age (years)",       min_value=18,    max_value=120,    value=60)
        weight  = st.number_input("Weight (kg)",       min_value=30.0,  max_value=300.0,  value=70.0)
        icu_los = st.number_input("ICU LOS (hours)",   min_value=0.0,   max_value=2000.0, value=48.0)
    with col2:
        gender  = st.selectbox("Gender",               ["Male", "Female"])
        height  = st.number_input("Height (cm)",       min_value=100.0, max_value=220.0,  value=170.0)
        mv_hrs  = st.number_input("MV Duration (hrs)", min_value=0.0,   max_value=2000.0, value=0.0)
    col1, col2 = st.columns(2)
    with col1:
        cancer = st.selectbox("Cancer", ["No", "Yes"])

    # ── Laboratory Values ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🧪 Laboratory Values</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pre_k   = st.number_input("Baseline K (mEq/L)",   min_value=1.0,  max_value=7.0,   value=3.2, step=0.1)
        dose    = st.number_input("KCl Dose (mEq)",        min_value=1.0,  max_value=200.0, value=40.0)
        gfr     = st.number_input("CrCl (mL/min)",         min_value=1.0,  max_value=200.0, value=80.0)
        mg      = st.number_input("Serum Mg (mEq/L)",      min_value=0.5,  max_value=5.0,   value=1.8, step=0.1)
        bicarb  = st.number_input("Bicarbonate (mEq/L)",   min_value=5.0,  max_value=50.0,  value=24.0)
    with col2:
        calcium = st.number_input("Calcium (mg/dL)",       min_value=5.0,  max_value=15.0,  value=9.0, step=0.1)
        glucose = st.number_input("Glucose (mg/dL)",       min_value=50.0, max_value=600.0, value=120.0)
        bun     = st.number_input("BUN (mg/dL)",           min_value=1.0,  max_value=200.0, value=20.0)
        albumin = st.number_input("Albumin (g/dL)",        min_value=1.0,  max_value=6.0,   value=3.5, step=0.1)
        creat   = st.number_input("Creatinine (mg/dL)",    min_value=0.1,  max_value=20.0,  value=1.0, step=0.1)

    # ── Medications ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💉 Medications & Interventions</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        mv          = st.checkbox("Mechanical Ventilation")
        loop        = st.checkbox("Loop Diuretics")
        thiazide    = st.checkbox("Thiazide Diuretics")
        calcineurin = st.checkbox("Calcineurin Inhibitors")
        digoxin     = st.checkbox("Digoxin")
        heparin     = st.checkbox("Heparin")
        magnesium   = st.checkbox("IV Magnesium")
        feeding     = st.checkbox("Enteral Feeding")
    with col2:
        glucocort   = st.checkbox("Glucocorticoids")
        beta        = st.checkbox("Beta-adrenergic Agents")
        insulin     = st.checkbox("Insulin")
        vasopressor = st.checkbox("Vasopressors")
        ace         = st.checkbox("ACE Inhibitors")
        sodium_bic  = st.checkbox("Sodium Bicarbonate")
        arbs        = st.checkbox("ARBs")
        potassium_s = st.checkbox("K-sparing Diuretics")

    # ── ICU Diagnosis ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏥 ICU Diagnosis</div>",
                unsafe_allow_html=True)
    icu_cat = st.selectbox("ICU Category", [
        "Cardiovascular", "Burns_trauma", "Endocrine",
        "Gastrointestinal", "Hematology", "Neurologic",
        "Other", "Renal", "Respiratory",
        "Sepsis_Septic_Shock", "Surgery", "Toxicology"
    ])

    # ── Feature engineering ───────────────────────────────────────────────
    gender_bin    = 1 if gender == "Male" else 0
    cancer_b      = int(cancer == "Yes")
    bmi           = weight / ((height / 100) ** 2)
    bmi_clean     = bmi if 10 <= bmi <= 100 else np.nan
    k_mild        = int(3.0 <= pre_k < 3.5)
    k_moderate    = int(2.5 <= pre_k < 3.0)
    k_severe      = int(pre_k < 2.5)
    gfr_3060      = int(30 <= gfr <= 60)
    gfr_lt30      = int(gfr < 30)
    mv_b          = int(mv);          loop_b        = int(loop)
    thiazide_b    = int(thiazide);    calcineurin_b = int(calcineurin)
    digoxin_b     = int(digoxin);     heparin_b     = int(heparin)
    magnesium_b   = int(magnesium);   glucocort_b   = int(glucocort)
    beta_b        = int(beta);        insulin_b     = int(insulin)
    vasopressor_b = int(vasopressor); ace_b         = int(ace)
    sodium_bic_b  = int(sodium_bic);  feeding_b     = int(feeding)

    icu_options = ['Burns_trauma','Endocrine','Gastrointestinal','Hematology',
                   'Neurologic','Other','Renal','Respiratory',
                   'Sepsis_Septic_Shock','Surgery','Toxicology']
    icu_dummies = {f"ICU_{c}": int(icu_cat == c) for c in icu_options}

    dose_per_kg      = dose / weight if weight > 0 else np.nan
    K_GFR_interact   = pre_k * gfr
    K_dose_interact  = pre_k * dose
    K_deficit        = 3.5 - pre_k
    K_deficit_dose   = K_deficit * dose
    loop_GFR         = loop_b * gfr
    Mg_K             = mg * pre_k
    bicarb_K         = bicarb * pre_k
    age_K_interact   = age * pre_k
    albumin_K        = albumin * pre_k
    creatinine_dose  = creat * dose
    glucose_K        = glucose * pre_k
    calcium_K        = calcium * pre_k
    BUN_creatinine   = bun / creat if creat > 0 else np.nan
    BMI_dose         = bmi_clean * dose if not np.isnan(bmi_clean) else np.nan
    MV_K             = mv_b * pre_k
    vasopressor_K    = vasopressor_b * pre_k
    insulin_K        = insulin_b * pre_k
    glucocorticoid_K = glucocort_b * pre_k
    cancer_K         = cancer_b * pre_k
    cancer_dose      = cancer_b * dose
    age_GFR          = age * gfr
    Mg_dose          = mg * dose
    K_severity_dose  = (k_mild + k_moderate * 2 + k_severe * 3) * dose

    input_dict = {
        'age': age, 'gender': gender_bin, 'BMI_clean': bmi_clean,
        'pre_labresult': pre_k, 'summed_dosage': dose, 'GFR': gfr,
        'pre_mg_level': mg, 'bicarbonate_lab': bicarb, 'calcium': calcium,
        'glucose': glucose, 'BUN': bun, 'albumin': albumin,
        'creatinine': creat, 'icu_los_hours': icu_los, 'hrs': mv_hrs,
        'MV_bin': mv_b, 'loop_bin': loop_b, 'thiazide_bin': thiazide_b,
        'Calcineurin_inhibitors_bin': calcineurin_b, 'digoxin_bin': digoxin_b,
        'heparin_bin': heparin_b, 'Magnesium_bin': magnesium_b,
        'Glucocorticoid_bin': glucocort_b, 'Beta_adrenergic_bin': beta_b,
        'Insulin_bin': insulin_b, 'vasopressor_bin': vasopressor_b,
        'ACE_inhabitor_bin': ace_b, 'sodium_bicarbonat_bin': sodium_bic_b,
        'feeding_bin': feeding_b, 'cancer_bin': cancer_b,
        'K_Mild': k_mild, 'K_Moderate': k_moderate, 'K_Severe': k_severe,
        'GFR_3060': gfr_3060, 'GFR_lt30': gfr_lt30,
        **icu_dummies,
        'dose_per_kg': dose_per_kg, 'K_GFR_interact': K_GFR_interact,
        'K_dose_interact': K_dose_interact, 'K_deficit': K_deficit,
        'K_deficit_dose': K_deficit_dose, 'loop_GFR': loop_GFR,
        'Mg_K': Mg_K, 'bicarb_K': bicarb_K,
        'age_K_interact': age_K_interact, 'albumin_K': albumin_K,
        'creatinine_dose': creatinine_dose, 'glucose_K': glucose_K,
        'calcium_K': calcium_K, 'BUN_creatinine': BUN_creatinine,
        'BMI_dose': BMI_dose, 'MV_K': MV_K,
        'vasopressor_K': vasopressor_K, 'insulin_K': insulin_K,
        'glucocorticoid_K': glucocorticoid_K, 'cancer_K': cancer_K,
        'cancer_dose': cancer_dose, 'age_GFR': age_GFR,
        'Mg_dose': Mg_dose, 'K_severity_dose': K_severity_dose,
    }

    input_df = pd.DataFrame([input_dict])

    # ── Predict ───────────────────────────────────────────────────────────
    predict_btn = st.button("🔮 Calculate Potassium Response")

    if predict_btn:
        try:
            bin_cat_cols = (
                [f+'_bin' for f in ['MV','loop','thiazide','Calcineurin_inhibitors',
                                    'digoxin','heparin','Magnesium','Glucocorticoid',
                                    'Beta_adrenergic','Insulin','vasopressor',
                                    'ACE_inhabitor','sodium_bicarbonat',
                                    'feeding','cancer']] +
                ['K_Mild','K_Moderate','K_Severe','GFR_3060','GFR_lt30'] +
                list(icu_dummies.keys())
            )
            cat_indices = [list(input_df.columns).index(c)
                           for c in bin_cat_cols if c in input_df.columns]

            pool        = Pool(input_df, cat_features=cat_indices)
            pred        = model.predict(pool)[0]
            total_delta = pred * dose / 10
            post_k      = pre_k + total_delta

            # ── Results ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Δ K per 10 mEq</div>
                        <div class='metric-value'>{pred:.3f}</div>
                        <div class='metric-unit'>mEq/L</div>
                    </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Total Expected Δ K</div>
                        <div class='metric-value'>{total_delta:.3f}</div>
                        <div class='metric-unit'>mEq/L</div>
                    </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Expected Post-dose K</div>
                        <div class='metric-value'>{post_k:.2f}</div>
                        <div class='metric-unit'>mEq/L</div>
                    </div>""", unsafe_allow_html=True)

            if post_k < 3.0:
                st.markdown(f"""
                    <div class='alert-warning'>
                        ⚠️ Expected post-dose K still low: {post_k:.2f} mEq/L<br>
                        Consider additional potassium replacement
                    </div>""", unsafe_allow_html=True)
            elif post_k > 5.5:
                st.markdown(f"""
                    <div class='alert-danger'>
                        🚨 Risk of hyperkalemia: {post_k:.2f} mEq/L<br>
                        Monitor closely — consider dose reduction
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='alert-success'>
                        ✅ Expected post-dose K in acceptable range: {post_k:.2f} mEq/L
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            severity = ("Normal"   if pre_k >= 3.5 else
                        "Mild"     if pre_k >= 3.0 else
                        "Moderate" if pre_k >= 2.5 else
                        "Severe")
            st.info(f"📌 Baseline K: **{severity}** ({pre_k} mEq/L) | "
                    f"Dose: **{dose} mEq** | Cancer: **{cancer}**")

            # ── Save to Google Sheets ──────────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-header'>💾 Save Patient Data</div>",
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input("Patient ID", value="")
            with col2:
                actual_k = st.number_input(
                    "Actual Post-dose K (mEq/L)",
                    min_value=1.0, max_value=9.0,
                    value=float(round(post_k, 1)), step=0.1)

            save_btn = st.button("💾 Save to Google Sheets")

            if save_btn:
                row = {
                    'timestamp':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'patient_id':             patient_id,
                    'age':                    age,
                    'gender':                 gender,
                    'cancer':                 cancer,
                    'icu_category':           icu_cat,
                    'baseline_K':             pre_k,
                    'dose_mEq':               dose,
                    'GFR':                    gfr,
                    'severity':               severity,
                    'predicted_delta_per_10': round(pred, 4),
                    'predicted_total_delta':  round(total_delta, 4),
                    'predicted_post_K':       round(post_k, 4),
                    'actual_post_K':          actual_k,
                    'error':                  round(actual_k - post_k, 4),
                    'abs_error':              round(abs(actual_k - post_k), 4),
                }
                if save_to_gsheet(row):
                    st.success(f"✅ Patient **{patient_id}** saved to Google Sheets permanently!")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — External Validation
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 External Validation":

    st.markdown("# 📊 External Validation")

    df_val = load_from_gsheet()

    if df_val.empty:
        st.warning("No data saved yet.")
    else:
        st.markdown(f"**Total patients saved: {len(df_val):,}**")
        df_complete = df_val.dropna(subset=['actual_post_K'])
        df_complete = df_complete[df_complete['actual_post_K'] > 0].copy()
        df_complete['actual_post_K']    = pd.to_numeric(df_complete['actual_post_K'],    errors='coerce')
        df_complete['predicted_post_K'] = pd.to_numeric(df_complete['predicted_post_K'], errors='coerce')
        df_complete = df_complete.dropna(subset=['actual_post_K','predicted_post_K'])

        if len(df_complete) < 5:
            st.info(f"{len(df_complete)} complete records. Need at least 5.")
        else:
            mae  = mean_absolute_error(df_complete['actual_post_K'],
                                       df_complete['predicted_post_K'])
            r2   = r2_score(df_complete['actual_post_K'],
                            df_complete['predicted_post_K'])
            bias = (df_complete['predicted_post_K'] -
                    df_complete['actual_post_K']).mean()
            rmse = np.sqrt(((df_complete['predicted_post_K'] -
                             df_complete['actual_post_K'])**2).mean())

            st.markdown("### 📈 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²",   f"{r2:.3f}")
            col2.metric("MAE",  f"{mae:.3f} mEq/L")
            col3.metric("RMSE", f"{rmse:.3f} mEq/L")
            col4.metric("Bias", f"{bias:.3f} mEq/L")

            st.markdown("---")

            # Predicted vs Actual
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(df_complete['actual_post_K'],
                        df_complete['predicted_post_K'],
                        alpha=0.6, color='#2C3E7A', s=50)
            mn = min(df_complete['actual_post_K'].min(),
                     df_complete['predicted_post_K'].min()) - 0.2
            mx = max(df_complete['actual_post_K'].max(),
                     df_complete['predicted_post_K'].max()) + 0.2
            ax1.plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect prediction')
            ax1.set_xlabel('Actual Post-dose K (mEq/L)')
            ax1.set_ylabel('Predicted Post-dose K (mEq/L)')
            ax1.set_title(f'Predicted vs Actual | N={len(df_complete)} | R²={r2:.3f}',
                          fontweight='bold')
            ax1.legend(); ax1.grid(True, alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            st.pyplot(fig1)

            # Bland-Altman
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            means     = (df_complete['actual_post_K'] +
                         df_complete['predicted_post_K']) / 2
            diffs     = df_complete['actual_post_K'] - df_complete['predicted_post_K']
            mean_diff = diffs.mean()
            std_diff  = diffs.std()
            ax3.scatter(means, diffs, alpha=0.6, color='#2C3E7A', s=50)
            ax3.axhline(mean_diff,               color='red',  lw=2,
                        label=f'Mean bias: {mean_diff:.3f}')
            ax3.axhline(mean_diff+1.96*std_diff, color='gray', lw=1.5,
                        linestyle='--', label=f'+1.96 SD: {mean_diff+1.96*std_diff:.3f}')
            ax3.axhline(mean_diff-1.96*std_diff, color='gray', lw=1.5,
                        linestyle='--', label=f'-1.96 SD: {mean_diff-1.96*std_diff:.3f}')
            ax3.axhline(0, color='black', lw=0.8, alpha=0.4)
            ax3.set_xlabel('Mean of Actual and Predicted (mEq/L)')
            ax3.set_ylabel('Actual − Predicted (mEq/L)')
            ax3.set_title('Bland-Altman Plot', fontweight='bold')
            ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            st.pyplot(fig3)

        # Table
        st.markdown("### 📋 Saved Patients")
        show_cols = ['timestamp','patient_id','baseline_K','dose_mEq',
                     'cancer','icu_category','predicted_post_K',
                     'actual_post_K','error','abs_error']
        show_cols = [c for c in show_cols if c in df_val.columns]
        st.dataframe(df_val[show_cols].sort_values(
            'timestamp', ascending=False).head(50),
            use_container_width=True)

        # Download
        st.download_button(
            label="⬇️ Download All Data as CSV",
            data=df_val.to_csv(index=False),
            file_name='patient_predictions.csv',
            mime='text/csv'
        )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.75rem;'>"
    "For clinical decision support only. Always use clinical judgment.<br>"
    "Model trained on eICU Collaborative Research Database."
    "</p>", unsafe_allow_html=True)
