import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import time
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from howso.engine import Trainee
    from howso.utilities import infer_feature_attributes
    HOWSO_AVAILABLE = True
except ImportError:
    HOWSO_AVAILABLE = False

st.set_page_config(
    page_title="Howso AI Demo",
    page_icon="🤔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .prediction-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .influential-row {
        padding: 0.5rem;
        border-left: 3px solid #667eea;
        background: #f0f2ff;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
    .stApp {
        max-width: 100%;
    }
    .highlight-blue {
        color: #667eea;
        font-weight: 600;
    }
    .howso-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .rf-badge {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLUMN_NAMES = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
FEATURE_NAMES = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
TARGET = 'mpg'

ORIGIN_MAP = {1: 'USA', 2: 'Europe', 3: 'Japan'}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            DATA_URL,
            delim_whitespace=True,
            names=COLUMN_NAMES,
            na_values='?',
        )
        df.drop(columns=['car_name'], inplace=True)
        df.dropna(inplace=True)
        df['origin'] = df['origin'].astype(int)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        demo_data = {
            'mpg': [18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0, 15.0],
            'cylinders': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            'displacement': [307, 350, 318, 304, 302, 429, 454, 440, 455, 390],
            'horsepower': [130, 165, 150, 150, 140, 198, 220, 215, 225, 190],
            'weight': [3504, 3693, 3436, 3433, 3449, 4341, 4354, 4312, 4425, 3850],
            'acceleration': [12.0, 11.5, 11.0, 12.0, 10.5, 10.0, 9.0, 8.5, 10.0, 8.5],
            'model_year': [70, 70, 70, 70, 70, 70, 70, 70, 70, 70],
            'origin': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        return pd.DataFrame(demo_data)

@st.cache_resource
def train_howso(_df):
    if not HOWSO_AVAILABLE:
        return None
    df = _df.copy()
    features = infer_feature_attributes(df)
    features[TARGET]['type'] = 'continuous'
    try:
        trainee = Trainee(features=features)
        trainee.train(df)
        return trainee
    except Exception as e:
        st.error(f"Howso training failed: {e}")
        return None

@st.cache_resource
def train_rf_model(_df):
    df = _df.copy()
    X = df[FEATURE_NAMES]
    y = df[TARGET]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def run_small_data_comparison(df, train_sizes):
    results = []
    for size in train_sizes:
        if size >= len(df):
            continue
        train = df.sample(n=size, random_state=42)
        test = df.drop(train.index)

        X_train, X_test = train[FEATURE_NAMES], test[FEATURE_NAMES]
        y_train, y_test = train[TARGET], test[TARGET]

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)

        howso_r2 = None
        howso_mae = None
        if HOWSO_AVAILABLE:
            try:
                features = infer_feature_attributes(train)
                features[TARGET]['type'] = 'continuous'
                t = Trainee(features=features)
                t.train(train)

                action_features = [TARGET]
                context_features = FEATURE_NAMES
                t.analyze(context_features=context_features, action_features=action_features)

                result = t.react(
                    contexts=X_test,
                    context_features=context_features,
                    action_features=action_features,
                )
                preds = result['action'][TARGET].values
                howso_r2 = r2_score(y_test, preds)
                howso_mae = mean_absolute_error(y_test, preds)
            except Exception:
                pass

        results.append({
            'train_size': size,
            'howso_r2': howso_r2,
            'howso_mae': howso_mae,
            'rf_r2': rf_r2,
            'rf_mae': rf_mae,
            'n_test': len(test),
        })
    return pd.DataFrame(results)

st.sidebar.markdown("""
<div style="text-align:center;margin-bottom:1rem;">
    <span style="font-size:2rem;">🤔</span>
    <h2 style="margin:0;color:#1a1a2e;">Howso AI</h2>
    <p style="font-size:0.8rem;color:#6c757d;">Explainable AI Demo</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "🎯 Predict & Explain",
    "📊 Small Data Showdown",
    "🔄 Synthetic Data Quality",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem;color:#6c757d;">
    Built with ❤️ using Howso Engine<br>
    Dataset: Auto MPG (UCI)
</div>
""", unsafe_allow_html=True)

if not HOWSO_AVAILABLE:
    st.sidebar.warning("⚠️ Howso Engine not installed. Some features require `pip install howso-engine`.")

df = load_data()

if page == "🎯 Predict & Explain":
    st.markdown("<h1 style='margin-bottom:0;'>🎯 Predict & Explain</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d;margin-top:0;'>Predict MPG with full attribution — see <span class='highlight-blue'>why</span> every prediction is made.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<div class='section-header'>Input Features</div>", unsafe_allow_html=True)

        cyl = st.slider("Cylinders", 3, 8, 6, help="Number of engine cylinders")
        disp = st.slider("Displacement (cu in)", 68, 455, 200, help="Engine displacement in cubic inches")
        hp = st.slider("Horsepower", 46, 230, 120, help="Engine horsepower")
        weight = st.slider("Weight (lbs)", 1613, 5140, 3000, help="Vehicle weight")
        accel = st.slider("Acceleration (0-60 mph)", 8.0, 24.8, 15.0, help="Time to reach 60 mph in seconds")
        year = st.slider("Model Year", 70, 82, 76, help="Model year of the vehicle")
        origin = st.selectbox("Origin", [1, 2, 3], format_func=lambda x: ORIGIN_MAP.get(x, x), help="1=USA, 2=Europe, 3=Japan")

        predict_btn = st.button("🚀 Predict MPG", type="primary", use_container_width=True)

    input_df = pd.DataFrame([{
        'cylinders': cyl, 'displacement': disp, 'horsepower': hp,
        'weight': weight, 'acceleration': accel, 'model_year': year, 'origin': origin,
    }])

    with col2:
        if not HOWSO_AVAILABLE:
            st.info("💡 Install Howso Engine to see explainable predictions: `pip install howso-engine`")

            rf_model = train_rf_model(df)
            rf_pred = rf_model.predict(input_df[FEATURE_NAMES])[0]

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-value">{rf_pred:.1f}</div>
                <div class="prediction-label">Predicted MPG (Random Forest)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-header'>📋 Feature Values</div>", unsafe_allow_html=True)
            st.dataframe(input_df, hide_index=True, use_container_width=True)

        else:
            with st.spinner("Training Howso model..."):
                trainee = train_howso(df)

            if trainee is None:
                st.error("Failed to initialize Howso. Check installation.")
            else:
                action_features = [TARGET]
                context_features = FEATURE_NAMES

                try:
                    trainee.analyze(
                        context_features=context_features,
                        action_features=action_features,
                    )
                except Exception:
                    pass

                if predict_btn:
                    with st.spinner("Generating prediction with full attribution..."):
                        try:
                            result = trainee.react(
                                contexts=input_df,
                                context_features=context_features,
                                action_features=action_features,
                                details={
                                    'influential_cases': True,
                                }
                            )
                            pred_value = result['action'][TARGET].iloc[0]

                            st.markdown(f"""
                            <div class="prediction-box">
                                <div class="prediction-value">{pred_value:.1f}</div>
                                <div class="prediction-label">Predicted MPG with Howso <span class="howso-badge">Attributed</span></div>
                            </div>
                            """, unsafe_allow_html=True)

                            if 'details' in result:
                                details = result['details']
                                st.markdown("<div class='section-header'>🔍 Influential Training Cases</div>", unsafe_allow_html=True)

                                infl_cases = None
                                if isinstance(details, dict):
                                    for key in ['influential_cases', 'influentialCases', 'influential cases']:
                                        if key in details:
                                            infl_cases = details[key]
                                            break

                                if infl_cases is not None:
                                    try:
                                        infl_df = pd.DataFrame(infl_cases)
                                        if not infl_df.empty:
                                            display_cols = [c for c in [TARGET] + FEATURE_NAMES if c in infl_df.columns]
                                            if 'weight' in infl_df.columns and 'influence_weight' not in infl_df.columns:
                                                pass
                                            st.dataframe(
                                                infl_df.head(10),
                                                hide_index=True,
                                                use_container_width=True,
                                            )
                                            st.caption(f"Top {min(10, len(infl_df))} most influential training cases")
                                    except Exception:
                                        st.write(infl_cases)
                                else:
                                    st.info("Influential cases detail was requested but not returned in expected format.")
                                    st.write(details)

                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            result = trainee.react(
                                contexts=input_df,
                                context_features=context_features,
                                action_features=action_features,
                            )
                            pred_value = result['action'][TARGET].iloc[0]
                            st.markdown(f"""
                            <div class="prediction-box">
                                <div class="prediction-value">{pred_value:.1f}</div>
                                <div class="prediction-label">Predicted MPG</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("👈 Adjust the inputs and click **Predict MPG** to see an explainable prediction with attribution.")

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Training Data Overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Samples", len(df))
    with c2:
        st.metric("Features", len(FEATURE_NAMES))
    with c3:
        st.metric("Avg MPG", f"{df[TARGET].mean():.1f}")
    with c4:
        st.metric("MPG Range", f"{df[TARGET].min():.0f} - {df[TARGET].max():.0f}")

elif page == "📊 Small Data Showdown":
    st.markdown("<h1 style='margin-bottom:0;'>📊 Small Data Showdown</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d;margin-top:0;'>Howso vs Random Forest — see how each performs when data is scarce.</p>", unsafe_allow_html=True)

    max_size = min(200, len(df))
    train_sizes = [10, 20, 30, 50, 75, 100, 150, max_size]
    train_sizes = [s for s in train_sizes if s <= max_size]

    with st.spinner("Running comparison across training sizes..."):
        results_df = run_small_data_comparison(df, train_sizes)

    if results_df.empty:
        st.warning("Not enough data for comparison.")
    else:
        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("<div class='section-header'>⚡ Key Insight</div>", unsafe_allow_html=True)

            small = results_df[results_df['train_size'] <= 30]
            if not small.empty:
                st.markdown("""
                <div style="background:#f0f2ff;padding:1rem;border-radius:10px;border-left:4px solid #667eea;">
                """, unsafe_allow_html=True)
                st.markdown("Howso excels on <b>small datasets</b> because its instance-based learning doesn't need massive amounts of data to generalize. Traditional models like Random Forest need more data to avoid overfitting.")
                st.markdown("</div>", unsafe_allow_html=True)

            best_howso = results_df.dropna(subset=['howso_r2']).head(1)
            if not best_howso.empty:
                st.metric(
                    "Howso R² (smallest data)",
                    f"{best_howso['howso_r2'].values[0]:.3f}",
                )

            best_rf_small = results_df[results_df['train_size'] <= 30]['rf_r2'].mean()
            st.metric("Avg RF R² (≤30 samples)", f"{best_rf_small:.3f}")

        with col1:
            fig = go.Figure()
            mask_h = results_df['howso_r2'].notna()
            if mask_h.any():
                fig.add_trace(go.Scatter(
                    x=results_df.loc[mask_h, 'train_size'],
                    y=results_df.loc[mask_h, 'howso_r2'],
                    mode='lines+markers',
                    name='Howso',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, symbol='diamond'),
                ))
            fig.add_trace(go.Scatter(
                x=results_df['train_size'],
                y=results_df['rf_r2'],
                mode='lines+markers',
                name='Random Forest',
                line=dict(color='#ff6b6b', width=3, dash='dash'),
                marker=dict(size=10, symbol='circle'),
            ))
            fig.update_layout(
                title='R² Score vs Training Set Size',
                xaxis_title='Training Samples',
                yaxis_title='R² Score',
                yaxis_range=[-0.5, 1.0],
                hovermode='x unified',
                legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99),
                template='plotly_white',
                height=450,
            )
            fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>📋 Detailed Results</div>", unsafe_allow_html=True)
        display = results_df.copy()
        display.columns = [
            'Train Size', 'Howso R²', 'Howso MAE', 'RF R²', 'RF MAE', 'Test Samples'
        ]
        st.dataframe(display, hide_index=True, use_container_width=True)

elif page == "🔄 Synthetic Data Quality":
    st.markdown("<h1 style='margin-bottom:0;'>🔄 Synthetic Data Quality</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d;margin-top:0;'>Generate synthetic data with Howso and compare fidelity against the original.</p>", unsafe_allow_html=True)

    if not HOWSO_AVAILABLE:
        st.info("💡 Install Howso Engine to generate synthetic data: `pip install howso-engine`")
        st.markdown("<div class='section-header'>📊 Original Data Summary</div>", unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
    else:
        with st.spinner("Generating synthetic data with Howso... (this may take a moment)"):
            try:
                features = infer_feature_attributes(df)
                features[TARGET]['type'] = 'continuous'

                trainee = Trainee(features=features)
                trainee.train(df)

                trainee.analyze()

                synth_result = trainee.react(
                    action_features=df.columns.tolist(),
                    desired_conviction=10,
                    generate_new_cases='always',
                    num_cases_to_generate=len(df),
                )
                synth_df = synth_result['action']

                for col in df.select_dtypes(include=['int64', 'int32']).columns:
                    if col in synth_df.columns:
                        try:
                            synth_df[col] = synth_df[col].round().astype(int)
                        except Exception:
                            pass

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='section-header'>📊 Original Data</div>", unsafe_allow_html=True)
                    st.dataframe(df.describe().round(2), use_container_width=True)
                with col2:
                    st.markdown("<div class='section-header'>📊 Synthetic Data</div>", unsafe_allow_html=True)
                    st.dataframe(synth_df.describe().round(2), use_container_width=True)

                st.markdown("<div class='section-header'>📈 Distribution Comparison</div>", unsafe_allow_html=True)
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cols_per_row = 3
                rows_needed = (len(num_cols) + cols_per_row - 1) // cols_per_row

                fig = make_subplots(
                    rows=rows_needed, cols=cols_per_row,
                    subplot_titles=num_cols,
                    vertical_spacing=0.12,
                    horizontal_spacing=0.08,
                )
                for i, col in enumerate(num_cols):
                    row = i // cols_per_row + 1
                    col_pos = i % cols_per_row + 1
                    fig.add_trace(
                        go.Histogram(x=df[col], name='Original', marker_color='#667eea', opacity=0.7, legendgroup='original', showlegend=(i == 0)),
                        row=row, col=col_pos,
                    )
                    if col in synth_df.columns:
                        fig.add_trace(
                            go.Histogram(x=synth_df[col], name='Synthetic', marker_color='#ff6b6b', opacity=0.6, legendgroup='synth', showlegend=(i == 0)),
                            row=row, col=col_pos,
                        )
                fig.update_layout(
                    height=200 * rows_needed,
                    barmode='overlay',
                    template='plotly_white',
                    hovermode='x unified',
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div class='section-header'>🔗 Correlation Comparison</div>", unsafe_allow_html=True)
                c1_h, c2_h = st.columns(2)
                with c1_h:
                    st.markdown("**Original Data**")
                    corr_orig = df[num_cols].corr()
                    fig_orig = px.imshow(
                        corr_orig,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect='auto',
                        height=400,
                    )
                    fig_orig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_orig, use_container_width=True)
                with c2_h:
                    st.markdown("**Synthetic Data**")
                    synth_num_cols = [c for c in num_cols if c in synth_df.columns]
                    corr_synth = synth_df[synth_num_cols].corr()
                    fig_synth = px.imshow(
                        corr_synth,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect='auto',
                        height=400,
                    )
                    fig_synth.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_synth, use_container_width=True)

                diff = (corr_orig - corr_synth.reindex_like(corr_orig)).abs()
                mean_diff = diff.values[np.triu_indices_from(diff.values, k=1)].mean() if len(diff) > 1 else 0

                st.markdown("<div class='section-header'>📊 Fidelity Metrics</div>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Mean Corr Diff", f"{mean_diff:.4f}", help="Average absolute correlation difference between original and synthetic")
                with m2:
                    orig_mean = df[num_cols].mean()
                    synth_mean = synth_df[[c for c in num_cols if c in synth_df.columns]].mean()
                    mean_bias = (orig_mean - synth_mean).abs().mean()
                    st.metric("Mean Feature Bias", f"{mean_bias:.3f}", help="Average absolute difference in feature means")
                with m3:
                    orig_std = df[num_cols].std()
                    synth_std = synth_df[[c for c in num_cols if c in synth_df.columns]].std()
                    std_bias = (orig_std - synth_std).abs().mean()
                    st.metric("Mean Std Diff", f"{std_bias:.3f}", help="Average absolute difference in feature standard deviations")
                with m4:
                    st.metric("Synthetic Samples", len(synth_df), help="Number of synthetic data points generated")

            except Exception as e:
                st.error(f"Synthetic data generation failed: {e}")
                st.markdown("""
                <div style="background:#fff3cd;padding:1rem;border-radius:10px;">
                    <b>💡 Tip:</b> Make sure Howso Engine is properly installed. 
                    The synthetic data feature requires a trained trainee with targetless analysis.
                </div>
                """, unsafe_allow_html=True)
