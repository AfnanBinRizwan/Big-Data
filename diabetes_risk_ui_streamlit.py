import io
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ------------------------------
# Constants / Feature Schema
# ------------------------------
FEATURES: List[str] = [
    'BMI',          # float (kg/m^2)
    'HighBP',       # 0/1
    'HighChol',     # 0/1
    'GenHlth',      # 1-5 (1=Excellent ... 5=Poor)
    'Age',          # numeric age in years (will be mapped if your pipeline expects bins)
    'PhysActivity', # 0/1 (1 = Yes)
    'DiffWalk',     # 0/1 (1 = Yes difficulty)
]

DEFAULT_THRESHOLD = 0.50
RECALL_FRIENDLY_THRESHOLD = 0.35  # suggested by many screening setups

st.set_page_config(page_title='Diabetes Risk Screening', page_icon='🩺', layout='centered')
st.title('🩺 Diabetes Risk Screening (Survey‑only)')
st.caption('Enter a few details to estimate your **risk probability** using your trained ML model.\n' \
          'This app assumes a scikit‑learn Pipeline saved as **.joblib** with the 7 survey‑style features.')

# ------------------------------
# Sidebar: Model loader
# ------------------------------
with st.sidebar:
    st.header('Model')

    # Option 1: Pick from local folder if user placed models under ./models
    model_dir = st.text_input('Model folder', value='models', help='Directory to scan for .joblib files')
    available: List[str] = []
    if os.path.isdir(model_dir):
        available = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    choice = st.selectbox('Select a model file', options=['(none)'] + available, index=0)

    uploaded = st.file_uploader('…or upload a .joblib file', type=['joblib'])

    model = None
    model_name: Optional[str] = None
    if uploaded is not None:
        # Load from uploaded buffer
        try:
            model = load(io.BytesIO(uploaded.read()))
            model_name = uploaded.name
            st.success(f'Loaded uploaded model: {model_name}')
        except Exception as e:
            st.error(f'Could not load uploaded model: {e}')
    elif choice != '(none)':
        path = os.path.join(model_dir, choice)
        try:
            model = load(path)
            model_name = choice
            st.success(f'Loaded model: {model_name}')
        except Exception as e:
            st.error(f'Could not load model from {path}: {e}')

    st.markdown('---')
    st.subheader('Decision threshold')
    threshold = st.slider('Positive when probability ≥', 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    st.caption(f'Tip: try **{RECALL_FRIENDLY_THRESHOLD:.2f}** to reduce false negatives (higher recall).')

    st.markdown('---')
    st.subheader('Batch scoring (optional)')
    st.caption('Upload a CSV with columns: ' + ', '.join([f'**{c}**' for c in FEATURES]))
    csv_file = st.file_uploader('Upload CSV for batch predictions', type=['csv'], key='csv')

# ------------------------------
# Helper: Build single-row dataframe
# ------------------------------

def to_df(
    BMI: float,
    HighBP: int,
    HighChol: int,
    GenHlth: int,
    Age: int,
    PhysActivity: int,
    DiffWalk: int,
) -> pd.DataFrame:
    row = {
        'BMI': float(BMI),
        'HighBP': int(HighBP),
        'HighChol': int(HighChol),
        'GenHlth': int(GenHlth),
        'Age': int(Age),
        'PhysActivity': int(PhysActivity),
        'DiffWalk': int(DiffWalk),
    }
    return pd.DataFrame([row], columns=FEATURES)

# ------------------------------
# Input form
# ------------------------------
with st.form('inputs'):
    st.subheader('Enter your details')

    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input('BMI (kg/m²)', min_value=10.0, max_value=70.0, value=27.5, step=0.1)
        genhlth = st.select_slider('General health (1=Excellent … 5=Poor)', options=[1,2,3,4,5], value=3)
        age = st.number_input('Age (years)', min_value=18, max_value=100, value=45, step=1)
    with col2:
        highbp = st.selectbox('High blood pressure (diagnosed)?', options=['No', 'Yes'], index=0)
        highchol = st.selectbox('High cholesterol (diagnosed)?', options=['No', 'Yes'], index=0)
        phys = st.selectbox('Physically active (past 30 days)?', options=['No', 'Yes'], index=1)
        diffwalk = st.selectbox('Difficulty walking or climbing stairs?', options=['No', 'Yes'], index=0)

    submitted = st.form_submit_button('Estimate my risk', use_container_width=True)

# ------------------------------
# Predict (single)
# ------------------------------
if submitted:
    if model is None:
        st.error('Please upload or select a trained model (.joblib) in the sidebar.')
    else:
        X = to_df(
            BMI=bmi,
            HighBP=1 if highbp == 'Yes' else 0,
            HighChol=1 if highchol == 'Yes' else 0,
            GenHlth=genhlth,
            Age=age,
            PhysActivity=1 if phys == 'Yes' else 0,
            DiffWalk=1 if diffwalk == 'Yes' else 0,
        )
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                # Handle binary or 3-class pipelines
                if proba.shape[1] == 2:
                    risk_prob = float(proba[0, 1])
                else:
                    # If 3-class (0=no,1=pre,2=diabetes), treat risk as P(class 2) or P(>=1)
                    p_pre = float(proba[0, 1])
                    p_diab = float(proba[0, 2])
                    risk_prob = float(p_diab)  # conservative: explicit diabetes
                yhat = int(risk_prob >= threshold)
            else:
                # Fall back to decision_function or predict
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(X).ravel()[0])
                    # Map score to probability-like via logistic for display
                    risk_prob = float(1 / (1 + np.exp(-score)))
                    yhat = int(risk_prob >= threshold)
                else:
                    yhat = int(model.predict(X)[0])
                    risk_prob = float(yhat)
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            risk_prob, yhat = None, None

        if risk_prob is not None:
            st.markdown('---')
            st.subheader('Result')
            meter = int(round(risk_prob * 100))
            st.metric('Estimated risk probability', f'{meter}%')

            if yhat == 1:
                st.error('⚠️ **High risk** at the chosen threshold.')
                st.caption('This does **not** diagnose diabetes. Consider clinical follow‑up and confirmatory testing.')
            else:
                st.success('✅ **Lower risk** at the chosen threshold.')
                st.caption('Maintain healthy habits; re‑assess periodically or if risk factors change.')

            with st.expander('Show model inputs'):
                st.dataframe(X, use_container_width=True)

# ------------------------------
# Batch scoring
# ------------------------------
if csv_file is not None and model is not None:
    try:
        df = pd.read_csv(csv_file)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f'Missing required columns: {missing}')
        else:
            probs = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(df)
                if probs.shape[1] == 2:
                    risk = probs[:, 1]
                else:
                    risk = probs[:, 2]
            else:
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(df)
                    risk = 1 / (1 + np.exp(-scores))
                else:
                    preds = model.predict(df)
                    risk = preds.astype(float)
            pred = (risk >= threshold).astype(int)
            out = df.copy()
            out['risk_probability'] = risk
            out['predicted_high_risk'] = pred
            st.subheader('Batch results')
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button('Download results CSV', data=out.to_csv(index=False), file_name='risk_predictions.csv')
    except Exception as e:
        st.error(f'Batch prediction failed: {e}')

# ------------------------------
# Explainability (global)
# ------------------------------
if model is not None:
    st.markdown('---')
    st.subheader('Model explainability (global)')
    explained = False

    # Coefficients (e.g., LogisticRegression inside Pipeline)
    try:
        est = model
        # If Pipeline, try to grab the final estimator
        if hasattr(model, 'named_steps'):
            est = list(model.named_steps.values())[-1]
        if hasattr(est, 'coef_'):
            coefs = np.ravel(est.coef_)
            # If multi-class, take coefficients for the positive/risk class (heuristic)
            if est.coef_.ndim > 1 and est.coef_.shape[0] > 1:
                # pick last row as "diabetes" class if trained with [0,1,2]
                coefs = est.coef_[-1]
            dfc = pd.DataFrame({'feature': FEATURES[:len(coefs)], 'weight': coefs})
            dfc = dfc.sort_values('weight', key=np.abs, ascending=False)
            st.bar_chart(dfc.set_index('feature'))
            st.caption('Absolute magnitude reflects influence in a linear model; sign shows direction (positive = higher risk).')
            explained = True
    except Exception:
        pass

    # Tree/boosting importances
    if not explained:
        try:
            est = model
            if hasattr(model, 'named_steps'):
                est = list(model.named_steps.values())[-1]
            if hasattr(est, 'feature_importances_'):
                imp = est.feature_importances_
                dfi = pd.DataFrame({'feature': FEATURES[:len(imp)], 'importance': imp})
                dfi = dfi.sort_values('importance', ascending=False)
                st.bar_chart(dfi.set_index('feature'))
                st.caption('Higher bar = greater contribution to splits in tree/boosting model.')
                explained = True
        except Exception:
            pass

    if not explained:
        st.info('This model does not expose coefficients or feature importances. For richer explanations, consider saving a Pipeline with an interpretable final estimator or add SHAP in your notebook.')

# ------------------------------
# Footer
# ------------------------------
st.markdown('---')
st.caption('This UI is for **screening** only and not a diagnostic device.\n' \
          'Always consult qualified health professionals for medical advice.')
