import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

try:
    from howso.engine import Trainee
    from howso.utilities import infer_feature_attributes
    HOWSO_AVAILABLE = True
except ImportError:
    HOWSO_AVAILABLE = False

LLM_KEYS = {}
for var in ["MISTRAL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
    val = os.environ.get(var)
    if val:
        LLM_KEYS[var] = val

for provider in ["Mistral", "OpenAI", "Anthropic"]:
    key = f"llm_key_{provider}"
    if key not in st.session_state:
        st.session_state[key] = ""

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "Mistral"

st.set_page_config(page_title="Howso AI Demo", layout="wide", initial_sidebar_state="expanded")

if "page" not in st.session_state:
    st.session_state.page = "Overview"

APP_CSS = """
<style>
    .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.5rem; }
    .card h3 { color: #4f46e5; margin: 0 0 0.5rem 0; }
    .card p { color: #6b7280; margin: 0; font-size: 0.9rem; }
    .prediction-box { background: linear-gradient(135deg, #4f46e5, #7c3aed); padding: 2rem; border-radius: 15px; text-align: center; }
    .prediction-value { font-size: 3.5rem; font-weight: 800; line-height: 1; color: white; }
    .prediction-label { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; color: rgba(255,255,255,0.9); }
    .section-header { font-size: 1.3rem; font-weight: 600; color: #1a1a2e; margin-top: 1rem; margin-bottom: 0.5rem; }
    .highlight-blue { color: #4f46e5; font-weight: 600; }
    .howso-badge { display: inline-block; background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .nav-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.2rem; text-align: center; }
    .nav-card-title { color: #1a1a2e; font-weight: 600; font-size: 1rem; }
    .nav-card-desc { color: #6b7280; font-size: 0.8rem; margin-top: 0.3rem; }
    .insight-box { background: #f0f2ff; border-left: 4px solid #4f46e5; padding: 1rem; border-radius: 0 8px 8px 0; }
    .divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
    .dataset-badge { color: #6b7280; font-size: 0.8rem; }
    button[kind="primary"] { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; border: none !important; }
</style>
"""

def nav_card(title, desc, page_name, col):
    with col:
        st.markdown(f"""
        <div class="nav-card" onclick="document.querySelector('[data-testid=\\'baseButton-header\\']').click()">
            <div class="nav-card-title">{title}</div>
            <div class="nav-card-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(title, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()

# ─── DATASETS ────────────────────────────────────────────────────────

MPG_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

DATASETS = {
    "MPG (Fuel Efficiency)": {
        "type": "regression",
        "target": "mpg",
        "features": ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"],
        "desc": "Predict fuel efficiency (MPG) from vehicle characteristics.",
        "industry": "Automotive",
    },
    "Diabetes (Health)": {
        "type": "classification",
        "target": "diabetes",
        "features": ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age"],
        "desc": "Predict diabetes onset risk from patient diagnostic data.",
        "industry": "Healthcare",
    },
    "Wine Quality (Food)": {
        "type": "regression",
        "target": "quality",
        "features": ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"],
        "desc": "Predict wine quality score from physicochemical properties.",
        "industry": "Food & Beverage",
    },
}

DIABETES_COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age", "diabetes"]

FEATURE_RANGES = {}

def get_feature_ranges(df, features):
    ranges = {}
    for f in features:
        if df[f].dtype in [np.int64, np.float64]:
            if df[f].nunique() <= 10:
                ranges[f] = {"values": sorted(df[f].unique()), "type": "categorical"}
            else:
                ranges[f] = {"min": float(df[f].min()), "max": float(df[f].max()), "type": "continuous"}
        else:
            uniq = sorted(df[f].unique())
            ranges[f] = {"values": uniq, "type": "categorical"}
    return ranges

@st.cache_data
def load_dataset(name):
    if name == "MPG (Fuel Efficiency)":
        cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
        try:
            df = pd.read_csv(MPG_URL, delim_whitespace=True, names=cols, na_values="?")
            df.drop(columns=["car_name"], inplace=True)
            df.dropna(inplace=True)
            df["origin"] = df["origin"].astype(int)
            return df
        except Exception:
            return pd.DataFrame()
    elif name == "Diabetes (Health)":
        try:
            df = pd.read_csv(DIABETES_URL, names=DIABETES_COLS)
            df = df[(df[["glucose", "blood_pressure", "bmi"]] != 0).all(axis=1)]
            df["diabetes"] = df["diabetes"].astype(int)
            return df
        except Exception:
            return pd.DataFrame()
    elif name == "Wine Quality (Food)":
        try:
            df = pd.read_csv(WINE_URL, sep=";")
            df["quality"] = df["quality"].astype(int)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ─── MODEL TRAINING ──────────────────────────────────────────────────

@st.cache_resource
def train_howso(_df, ds_name):
    if not HOWSO_AVAILABLE:
        return None
    info = DATASETS[ds_name]
    target = info["target"]
    features_list = info["features"]
    df = _df.copy()
    try:
        features = infer_feature_attributes(df)
        features[target]["type"] = "continuous" if info["type"] == "regression" else "nominal"
        for f in features_list:
            if f in df.select_dtypes(["int64", "int32", "int8"]).columns and f != target:
                if df[f].nunique() <= 10:
                    features[f]["type"] = "nominal"
        trainee = Trainee(features=features)
        trainee.train(df)
        trainee.analyze(context_features=features_list, action_features=[target])
        return trainee
    except Exception as e:
        st.error(f"Howso training failed: {e}")
        return None

@st.cache_resource
def train_sklearn(_df, ds_name):
    info = DATASETS[ds_name]
    target = info["target"]
    features_list = info["features"]
    df = _df.copy()
    X = df[features_list]
    y = df[target]
    if info["type"] == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

@st.cache_data
def run_small_data_compare(_df, ds_name):
    info = DATASETS[ds_name]
    target = info["target"]
    features_list = info["features"]
    df = _df.copy()
    max_size = min(200, len(df))
    sizes = [s for s in [10, 20, 30, 50, 75, 100, 150, max_size] if s <= max_size]
    results = []
    for size in sizes:
        if size >= len(df):
            continue
        train = df.sample(n=size, random_state=42)
        test = df.drop(train.index)
        X_train, X_test = train[features_list], test[features_list]
        y_train, y_test = train[target], test[target]

        if info["type"] == "regression":
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_score = r2_score(y_test, rf_pred) if info["type"] == "regression" else accuracy_score(y_test, rf_pred)

        howso_score = None
        if HOWSO_AVAILABLE:
            try:
                feats = infer_feature_attributes(train)
                feats[target]["type"] = "continuous" if info["type"] == "regression" else "nominal"
                t = Trainee(features=feats)
                t.train(train)
                t.analyze(context_features=features_list, action_features=[target])
                res = t.react(contexts=X_test, context_features=features_list, action_features=[target])
                preds = res["action"][target].values
                if info["type"] == "regression":
                    howso_score = r2_score(y_test, preds)
                else:
                    howso_score = accuracy_score(y_test, preds.round())
            except Exception:
                pass
        results.append({"train_size": size, "howso_score": howso_score, "rf_score": rf_score, "n_test": len(test)})
    return pd.DataFrame(results)

# ─── MISTRAL ─────────────────────────────────────────────────────────

def get_llm_key(provider=None):
    if provider is None:
        provider = st.session_state.get("llm_provider", "Mistral")
    session_key = st.session_state.get(f"llm_key_{provider}", "")
    if session_key:
        return session_key
    env_key = f"{provider.upper()}_API_KEY"
    if env_key in LLM_KEYS:
        return LLM_KEYS[env_key]
    try:
        return st.secrets[env_key]
    except Exception:
        pass
    return None

def build_llm_prompt(features_dict, dataset_info):
    target = dataset_info["target"]
    task_type = dataset_info["type"]
    feat_str = ", ".join(f"{k}={v}" for k, v in features_dict.items())
    if task_type == "regression":
        return f"Given these features: {feat_str}, predict the {target}. Respond with ONLY a single numeric value, nothing else."
    else:
        return f"Given these features: {feat_str}, predict the {target} (0 or 1). Respond with ONLY 0 or 1, nothing else."

def query_llm(features_dict, dataset_info):
    provider = st.session_state.get("llm_provider", "Mistral")
    key = get_llm_key(provider)
    if not key:
        return None, f"No {provider} key configured"

    prompt = build_llm_prompt(features_dict, dataset_info)

    try:
        if provider == "Mistral":
            from mistralai.client import Mistral
            client = Mistral(api_key=key)
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=10,
            )
            text = response.choices[0].message.content.strip()

        elif provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=10,
            )
            text = response.choices[0].message.content.strip()

        elif provider == "Anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=key)
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

        else:
            return None, f"Unknown provider: {provider}"

        try:
            val = float(text.split()[0].replace(",", ""))
            return val, None
        except ValueError:
            return None, f"Could not parse response: {text}"

    except Exception as e:
        return None, str(e)

# ─── SIDEBAR ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div style='text-align:center;margin-bottom:1rem;'><h2 style='margin:0;'>Howso AI</h2><p style='font-size:0.8rem;opacity:0.6;'>Explainable AI Demo</p></div>", unsafe_allow_html=True)

    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    nav_items = [
        ("Overview", "Home"),
        ("Predict & Analyze", "Predict"),
        ("Howso vs LLM", "Compare"),
        ("Small Data", "SmallData"),
        ("Synthetic Data", "Synthetic"),
    ]
    for label, key in nav_items:
        if st.button(label, key=f"side_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;opacity:0.5;'>Built with Howso Engine</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.75rem;font-weight:600;margin-top:0.5rem;'>LLM Provider</div>", unsafe_allow_html=True)
    provider = st.selectbox("Provider", ["Mistral", "OpenAI", "Anthropic"], key="llm_provider_sel", label_visibility="collapsed")
    st.session_state.llm_provider = provider

    env_key_name = f"{provider.upper()}_API_KEY"
    stored_key = LLM_KEYS.get(env_key_name, "")
    session_key = st.session_state.get(f"llm_key_{provider}", "")
    active_key = stored_key or session_key

    if active_key:
        st.markdown(f"<div style='font-size:0.75rem;color:#10b981;'>{provider}: Connected</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-size:0.75rem;opacity:0.5;'>{provider}: No key</div>", unsafe_allow_html=True)
        entered = st.text_input(f"{provider} API Key", type="password", key=f"llm_input_{provider}", label_visibility="collapsed", placeholder=f"Enter {provider} key...")
        if entered:
            st.session_state[f"llm_key_{provider}"] = entered
            st.rerun()

    if not HOWSO_AVAILABLE:
        st.warning("Howso Engine not installed. Some features require `pip install howso-engine`.", icon="")

# ─── PAGES ──────────────────────────────────────────────────────────

df = load_dataset("MPG (Fuel Efficiency)")

if st.session_state.page == "Overview":
    st.markdown("<h1>Howso AI &mdash; Explainable Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem;opacity:0.7;margin-bottom:2rem;'>Howso is the only AI engine that provides <strong>full attribution</strong> for every prediction. Unlike black-box models, Howso shows you exactly which data points drove each decision &mdash; not just <em>what</em> it predicted, but <strong>why</strong>.</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='margin-bottom:1rem;'>Explore the Demo</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <h3>Predict & Analyze</h3>
            <p>Make predictions with full attribution across 3 real-world datasets. See which training cases influenced each result.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Predict & Analyze", key="home_predict", use_container_width=True):
            st.session_state.page = "Predict"
            st.rerun()
    with c2:
        st.markdown("""
        <div class="card">
            <h3>Howso vs LLM</h3>
            <p>Compare Howso's transparent, attributable predictions against a black-box LLM. See why attribution matters.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Howso vs LLM", key="home_compare", use_container_width=True):
            st.session_state.page = "Compare"
            st.rerun()
    with c3:
        st.markdown("""
        <div class="card">
            <h3>Small Data + Synthetic</h3>
            <p>See Howso outperform traditional models when data is scarce, and generate privacy-preserving synthetic data.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom:1rem;'>Why Attribution Matters</h3>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="padding:1rem;">
            <h4 style="color:#ef4444;">Black-Box LLM</h4>
            <ul style="opacity:0.7;">
                <li>You get an answer, but no <em>why</em></li>
                <li>Cannot audit or debug decisions</li>
                <li>Regulatory risk (GDPR, FCRA, HIPAA)</li>
                <li>Hidden bias and hallucinations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="padding:1rem;">
            <h4 style="color:#10b981;">Howso Attribution</h4>
            <ul style="opacity:0.7;">
                <li>Every prediction traces to specific training data</li>
                <li>Fully auditable and explainable</li>
                <li>Compliant by design</li>
                <li>Built-in bias detection and mitigation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "Predict":
    st.markdown("<h1>Predict & Analyze</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.6;margin-bottom:1.5rem;'>Select a dataset, adjust inputs, and see Howso's attributable predictions in action.</p>", unsafe_allow_html=True)

    ds_name = st.selectbox("Dataset", list(DATASETS.keys()), key="pred_ds")
    info = DATASETS[ds_name]
    df = load_dataset(ds_name)

    if df.empty:
        st.error(f"Could not load dataset: {ds_name}")
        st.stop()

    ranges = get_feature_ranges(df, info["features"])

    col_in, col_out = st.columns([1, 1.5])

    with col_in:
        st.markdown(f"<div class='section-header'>Input Features</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='dataset-badge'>{info['industry']} &middot; {info['type']}</span>", unsafe_allow_html=True)

        input_vals = {}
        for feat in info["features"]:
            r = ranges.get(feat, {})
            if r.get("type") == "categorical":
                input_vals[feat] = st.selectbox(feat.replace("_", " ").title(), r["values"], key=f"in_{feat}")
            else:
                mid = (r["min"] + r["max"]) / 2
                input_vals[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=float(r["min"]),
                    max_value=float(r["max"]),
                    value=float(mid),
                    key=f"in_{feat}",
                )

        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    input_df = pd.DataFrame([input_vals])

    with col_out:
        if predict_btn:
            if not HOWSO_AVAILABLE:
                st.info("Install Howso Engine: `pip install howso-engine`")
                model = train_sklearn(df, ds_name)
                pred = model.predict(input_df)[0]
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-value">{pred:.2f}</div>
                    <div class="prediction-label">Predicted {info['target']} (Random Forest)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                trainee = train_howso(df, ds_name)
                if trainee:
                    try:
                        result = trainee.react(
                            contexts=input_df,
                            context_features=info["features"],
                            action_features=[info["target"]],
                            details={"influential_cases": True},
                        )
                        pred_val = result["action"][info["target"]].iloc[0]
                        if info["type"] == "classification":
                            pred_val = int(round(pred_val))

                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-value">{pred_val}</div>
                            <div class="prediction-label">Predicted {info['target']} <span class="howso-badge">Attributed</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                        if "details" in result:
                            det = result["details"]
                            infl = None
                            if isinstance(det, dict):
                                for k in ["influential_cases", "influentialCases"]:
                                    if k in det:
                                        infl = det[k]
                                        break
                            if infl is not None:
                                st.markdown("<div class='section-header'>Influential Training Cases</div>", unsafe_allow_html=True)
                                try:
                                    st.dataframe(pd.DataFrame(infl).head(8), hide_index=True, use_container_width=True)
                                except Exception:
                                    with st.expander("Raw attribution data"):
                                        st.write(infl)

                        st.markdown("<div class='section-header'>Feature Values</div>", unsafe_allow_html=True)
                        st.dataframe(input_df, hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        result = trainee.react(contexts=input_df, context_features=info["features"], action_features=[info["target"]])
                        pred_val = result["action"][info["target"]].iloc[0]
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-value">{pred_val:.2f}</div>
                            <div class="prediction-label">Predicted {info['target']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Howso model failed to train.")
        else:
            st.info("Adjust inputs and click Predict.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Training Data</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", len(df))
    with c2:
        st.metric("Features", len(info["features"]))
    with c3:
        st.metric("Task Type", info["type"].title())
    with c4:
        if info["type"] == "regression":
            st.metric(f"Avg {info['target']}", f"{df[info['target']].mean():.2f}")
        else:
            st.metric("Class Balance", f"{df[info['target']].mean()*100:.0f}% positive")

elif st.session_state.page == "Compare":
    st.markdown("<h1>Howso vs LLM</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.6;margin-bottom:1.5rem;'>See the critical difference between attributable and black-box predictions. Howso explains <em>why</em>; LLMs just answer.</p>", unsafe_allow_html=True)

    ds_name = st.selectbox("Dataset", list(DATASETS.keys()), key="comp_ds")
    info = DATASETS[ds_name]
    df = load_dataset(ds_name)

    if df.empty:
        st.error(f"Could not load dataset: {ds_name}")
        st.stop()

    ranges = get_feature_ranges(df, info["features"])

    col_in, col_out = st.columns([1, 1.5])

    with col_in:
        st.markdown("<div class='section-header'>Input Features</div>", unsafe_allow_html=True)
        input_vals = {}
        for feat in info["features"]:
            r = ranges.get(feat, {})
            if r.get("type") == "categorical":
                input_vals[feat] = st.selectbox(feat.replace("_", " ").title(), r["values"], key=f"cmp_{feat}")
            else:
                mid = (r["min"] + r["max"]) / 2
                input_vals[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=float(r["min"]),
                    max_value=float(r["max"]),
                    value=float(mid),
                    key=f"cmp_{feat}",
                )
        compare_btn = st.button("Compare", type="primary", use_container_width=True)

    input_df = pd.DataFrame([input_vals])

    with col_out:
        if compare_btn:
            howso_pred = None
            howso_attribution = None
            mistral_pred = None
            mistral_error = None

            if HOWSO_AVAILABLE:
                trainee = train_howso(df, ds_name)
                if trainee:
                    try:
                        result = trainee.react(
                            contexts=input_df,
                            context_features=info["features"],
                            action_features=[info["target"]],
                            details={"influential_cases": True},
                        )
                        howso_pred = result["action"][info["target"]].iloc[0]
                        howso_attribution = result.get("details", {})
                    except Exception as e:
                        howso_pred = f"Error: {e}"

            mistral_pred, mistral_error = query_llm(input_vals, info)

            c_h, c_m = st.columns(2)
            with c_h:
                st.markdown("<div style='text-align:center;padding:1rem;background:#10b98110;border-radius:12px;border:1px solid #10b98130;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#10b981;margin:0;'>Howso</h3>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem;opacity:0.6;'>Attributable</p>", unsafe_allow_html=True)
                if howso_pred is not None:
                    val = howso_pred
                    if isinstance(val, float):
                        st.markdown(f"<div style='font-size:2.5rem;font-weight:800;'>{val:.2f}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='font-size:2.5rem;font-weight:800;'>{val}</div>", unsafe_allow_html=True)
                    st.markdown("<div style='font-size:0.8rem;opacity:0.6;margin-top:0.5rem;'>Prediction traces to specific training data</div>", unsafe_allow_html=True)

                    if howso_attribution and isinstance(howso_attribution, dict):
                        infl = None
                        for k in ["influential_cases", "influentialCases"]:
                            if k in howso_attribution:
                                infl = howso_attribution[k]
                                break
                        if infl is not None:
                            st.markdown("<div style='font-size:0.8rem;font-weight:600;margin-top:1rem;'>Top Influential Cases</div>", unsafe_allow_html=True)
                            try:
                                st.dataframe(pd.DataFrame(infl).head(4), hide_index=True, use_container_width=True)
                            except Exception:
                                pass
                else:
                    st.markdown("<div style='font-size:1rem;opacity:0.5;'>Not available</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            provider_name = st.session_state.get("llm_provider", "Mistral")
            with c_m:
                st.markdown("<div style='text-align:center;padding:1rem;background:#ef444410;border-radius:12px;border:1px solid #ef444430;'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:#ef4444;margin:0;'>{provider_name}</h3>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem;opacity:0.6;'>Black-Box LLM</p>", unsafe_allow_html=True)
                if mistral_pred is not None:
                    st.markdown(f"<div style='font-size:2.5rem;font-weight:800;'>{mistral_pred:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div style='font-size:0.8rem;opacity:0.6;margin-top:0.5rem;'>No attribution available. Cannot trace or audit.</div>", unsafe_allow_html=True)
                elif mistral_error:
                    st.markdown(f"<div style='font-size:0.9rem;color:#ef4444;'>{mistral_error}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size:1rem;opacity:0.5;'>Configure {provider_name} key in sidebar</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='insight-box' style='margin-top:1rem;'>", unsafe_allow_html=True)
            if howso_pred is not None and mistral_pred is not None:
                st.markdown(f"Both models made a prediction, but only Howso can show you <strong>why</strong>. The prediction is auditable, traceable, and backed by specific training data. {provider_name} gives you an answer with no way to verify, debug, or trust it.")
            else:
                st.markdown("When both models return results, you'll see the critical difference: Howso attributes every prediction to specific data points. LLMs provide no traceability.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Adjust inputs and click Compare.")

elif st.session_state.page == "SmallData":
    st.markdown("<h1>Small Data Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.6;margin-bottom:1.5rem;'>Howso excels where traditional models struggle &mdash; when data is limited. Instance-based learning generalizes from fewer examples.</p>", unsafe_allow_html=True)

    ds_name = st.selectbox("Dataset", list(DATASETS.keys()), key="sd_ds")
    df = load_dataset(ds_name)
    if df.empty:
        st.error(f"Could not load dataset: {ds_name}")
        st.stop()

    with st.spinner("Running comparison..."):
        results = run_small_data_compare(df, ds_name)

    if results.empty:
        st.warning("Not enough data.")
    else:
        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            fig = go.Figure()
            mask = results["howso_score"].notna()
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=results.loc[mask, "train_size"], y=results.loc[mask, "howso_score"],
                    mode="lines+markers", name="Howso",
                    line=dict(color="#10b981", width=3), marker=dict(size=10, symbol="diamond"),
                ))
            fig.add_trace(go.Scatter(
                x=results["train_size"], y=results["rf_score"],
                mode="lines+markers", name="Random Forest",
                line=dict(color="#ef4444", width=3, dash="dash"), marker=dict(size=10),
            ))
            metric_label = "R Score" if DATASETS[ds_name]["type"] == "regression" else "Accuracy"
            fig.update_layout(
                title=f"{metric_label} vs Training Size",
                xaxis_title="Training Samples", yaxis_title=metric_label,
                yaxis_range=[-0.5, 1.0], hovermode="x unified",
                template="plotly_white",
                height=450,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("<div class='section-header'>Key Insight</div>", unsafe_allow_html=True)
            small = results[results["train_size"] <= 30]
            if not small.empty:
                avg_rf = small["rf_score"].mean()
                avg_h = small["howso_score"].dropna().mean()
                st.markdown(f"""
                <div class='insight-box'>
                Howso leverages each data point directly through instance-based learning. With limited data, it extracts more signal than traditional models that need bulk statistics to generalize.
                </div>
                """, unsafe_allow_html=True)
                if not pd.isna(avg_h):
                    st.metric("Howso avg (<=30)", f"{avg_h:.3f}")
                st.metric("RF avg (<=30)", f"{avg_rf:.3f}")

        st.markdown("<div class='section-header'>Detailed Results</div>", unsafe_allow_html=True)
        display = results.copy()
        display.columns = ["Train Size", "Howso Score", "RF Score", "Test Samples"]
        st.dataframe(display, hide_index=True, use_container_width=True)

elif st.session_state.page == "Synthetic":
    st.markdown("<h1>Synthetic Data Quality</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.6;margin-bottom:1.5rem;'>Generate privacy-preserving synthetic data with Howso and compare statistical fidelity against the original.</p>", unsafe_allow_html=True)

    if not HOWSO_AVAILABLE:
        st.info("Install Howso Engine: `pip install howso-engine`")
        st.stop()

    ds_name = st.selectbox("Dataset", list(DATASETS.keys()), key="syn_ds")
    df = load_dataset(ds_name)
    if df.empty:
        st.error(f"Could not load dataset: {ds_name}")
        st.stop()

    with st.spinner("Generating synthetic data..."):
        try:
            features = infer_feature_attributes(df)
            info = DATASETS[ds_name]
            target = info["target"]
            features[target]["type"] = "continuous" if info["type"] == "regression" else "nominal"
            for f in info["features"]:
                if f in df.select_dtypes(["int64", "int32"]).columns and df[f].nunique() <= 10:
                    features[f]["type"] = "nominal"

            trainee = Trainee(features=features)
            trainee.train(df)
            trainee.analyze()

            synth = trainee.react(
                action_features=df.columns.tolist(),
                desired_conviction=10,
                generate_new_cases="always",
                num_cases_to_generate=len(df),
            )
            synth_df = synth["action"]

            for col in df.select_dtypes(["int64", "int32"]).columns:
                if col in synth_df.columns:
                    try:
                        synth_df[col] = synth_df[col].round().astype(int)
                    except Exception:
                        pass

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-header'>Original</div>", unsafe_allow_html=True)
                st.dataframe(df.describe().round(2), use_container_width=True)
            with c2:
                st.markdown("<div class='section-header'>Synthetic</div>", unsafe_allow_html=True)
                st.dataframe(synth_df.describe().round(2), use_container_width=True)

            num_cols = df.select_dtypes([np.number]).columns.tolist()
            st.markdown("<div class='section-header'>Distribution Comparison</div>", unsafe_allow_html=True)
            n_per_row = 3
            rows = (len(num_cols) + n_per_row - 1) // n_per_row
            fig = make_subplots(rows=rows, cols=n_per_row, subplot_titles=num_cols, vertical_spacing=0.12, horizontal_spacing=0.08)
            for i, col in enumerate(num_cols):
                r = i // n_per_row + 1
                c = i % n_per_row + 1
                fig.add_trace(go.Histogram(x=df[col], name="Original", marker_color="#6366f1", opacity=0.7, legendgroup="orig", showlegend=(i == 0)), row=r, col=c)
                if col in synth_df.columns:
                    fig.add_trace(go.Histogram(x=synth_df[col], name="Synthetic", marker_color="#ef4444", opacity=0.5, legendgroup="synth", showlegend=(i == 0)), row=r, col=c)
            fig.update_layout(height=200 * rows, barmode="overlay", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='section-header'>Correlation Comparison</div>", unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                corr_o = df[num_cols].corr()
                fig_o = px.imshow(corr_o, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", height=380)
                fig_o.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_o, use_container_width=True)
            with cb:
                sn = [c for c in num_cols if c in synth_df.columns]
                corr_s = synth_df[sn].corr()
                fig_s = px.imshow(corr_s, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", height=380)
                fig_s.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_s, use_container_width=True)

            diff = (corr_o - corr_s.reindex_like(corr_o)).abs()
            triu = np.triu_indices_from(diff.values, k=1)
            mean_diff = diff.values[triu].mean() if len(diff) > 1 and triu[0].size > 0 else 0
            orig_mean = df[num_cols].mean()
            syn_mean = synth_df[[c for c in num_cols if c in synth_df.columns]].mean()
            orig_std = df[num_cols].std()
            syn_std = synth_df[[c for c in num_cols if c in synth_df.columns]].std()

            st.markdown("<div class='section-header'>Fidelity Metrics</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean Corr Diff", f"{mean_diff:.4f}")
            m2.metric("Mean Bias", f"{(orig_mean - syn_mean).abs().mean():.3f}")
            m3.metric("Std Diff", f"{(orig_std - syn_std).abs().mean():.3f}")
            m4.metric("Synthetic Samples", len(synth_df))

        except Exception as e:
            st.error(f"Synthetic data generation failed: {e}")
