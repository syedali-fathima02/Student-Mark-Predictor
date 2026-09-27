import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page Configuration
st.set_page_config(
    page_title="Multiple Linear Regression - Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ALL session_state keys at the top of the app before any widget uses them
if "hours_studied" not in st.session_state:
    st.session_state.hours_studied = 6.5
if "sleep_hours" not in st.session_state:
    st.session_state.sleep_hours = 7.0
if "attendance" not in st.session_state:
    st.session_state.attendance = 80.0
if "previous_marks" not in st.session_state:
    st.session_state.previous_marks = 70.0

if "slider_hours" not in st.session_state:
    st.session_state.slider_hours = 6.5
if "num_hours" not in st.session_state:
    st.session_state.num_hours = 6.5

if "slider_sleep" not in st.session_state:
    st.session_state.slider_sleep = 7.0
if "num_sleep" not in st.session_state:
    st.session_state.num_sleep = 7.0

if "slider_att" not in st.session_state:
    st.session_state.slider_att = 80.0
if "num_att" not in st.session_state:
    st.session_state.num_att = 80.0

if "slider_prev" not in st.session_state:
    st.session_state.slider_prev = 70.0
if "num_prev" not in st.session_state:
    st.session_state.num_prev = 70.0

# Synchronization Callbacks for Sliders & Number Inputs
def sync_hours_slider():
    val = float(st.session_state.slider_hours)
    st.session_state.hours_studied = val
    st.session_state.num_hours = val

def sync_hours_num():
    val = float(min(12.0, max(0.0, st.session_state.num_hours)))
    st.session_state.hours_studied = val
    st.session_state.slider_hours = val

def sync_sleep_slider():
    val = float(st.session_state.slider_sleep)
    st.session_state.sleep_hours = val
    st.session_state.num_sleep = val

def sync_sleep_num():
    val = float(min(12.0, max(0.0, st.session_state.num_sleep)))
    st.session_state.sleep_hours = val
    st.session_state.slider_sleep = val

def sync_att_slider():
    val = float(st.session_state.slider_att)
    st.session_state.attendance = val
    st.session_state.num_att = val

def sync_att_num():
    val = float(min(100.0, max(0.0, st.session_state.num_att)))
    st.session_state.attendance = val
    st.session_state.slider_att = val

def sync_prev_slider():
    val = float(st.session_state.slider_prev)
    st.session_state.previous_marks = val
    st.session_state.num_prev = val

def sync_prev_num():
    val = float(min(100.0, max(0.0, st.session_state.num_prev)))
    st.session_state.previous_marks = val
    st.session_state.slider_prev = val

# Custom CSS for modern glassmorphism design
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5);
        margin-bottom: 20px;
    }
    .prediction-score {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .grade-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.25);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    .badge-sample {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

FEATURES = ['Hours Studied', 'Sleep Hours', 'Attendance Percentage', 'Previous Marks']
TARGET = 'Marks Scored'

# Helper functions
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "student_scores.csv")
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        # Generate 100 sample rows if missing
        np.random.seed(42)
        h = np.round(np.random.uniform(1.0, 10.0, 100), 1)
        s = np.round(np.random.uniform(4.0, 9.5, 100), 1)
        a = np.round(np.random.uniform(55.0, 100.0, 100), 1)
        p = np.round(np.random.uniform(40.0, 98.0, 100), 1)
        m = np.round(np.clip(4.25*h + 1.15*s + 0.28*a + 0.32*p - 18.5 + np.random.normal(0, 3.2, 100), 10, 100), 1)
        df_gen = pd.DataFrame({'Hours Studied': h, 'Sleep Hours': s, 'Attendance Percentage': a, 'Previous Marks': p, 'Marks Scored': m})
        df_gen.to_csv(data_path, index=False)
        return df_gen
    return pd.read_csv(data_path)

def get_performance_grade(score):
    if score >= 90:
        return "A+ (Outstanding)"
    elif score >= 80:
        return "A (Excellent)"
    elif score >= 70:
        return "B (Good)"
    elif score >= 60:
        return "C (Satisfactory)"
    elif score >= 50:
        return "D (Pass)"
    else:
        return "F (Needs Improvement)"

# Load Dataset & Train Models
df = load_data()

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Simple Linear Regression (1 feature: Hours Studied)
simple_model = LinearRegression()
simple_model.fit(X_train[['Hours Studied']], y_train)
y_pred_simple = simple_model.predict(X_test[['Hours Studied']])
r2_simple = r2_score(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

# Model 2: Multiple Linear Regression (All 4 features)
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_multi = multi_model.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)

# Standardized Feature Importance Analysis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)

raw_coefs = multi_model.coef_
beta_coefs = scaled_model.coef_
intercept = multi_model.intercept_

impact_df = pd.DataFrame({
    'Feature': FEATURES,
    'Raw Coef (w)': raw_coefs,
    'Standardized Beta': beta_coefs,
    'Abs Beta': np.abs(beta_coefs)
}).sort_values(by='Abs Beta', ascending=False)
impact_df['Impact %'] = (impact_df['Abs Beta'] / impact_df['Abs Beta'].sum()) * 100

top_feature_row = impact_df.iloc[0]

# Sidebar Controls for 4 Features
st.sidebar.title("🎓 Predictor Controls")
st.sidebar.markdown("Adjust student input features to predict final exam marks in real-time.")

# Feature 1: Hours Studied
st.sidebar.subheader("1. 📖 Hours Studied")
st.sidebar.slider(
    "Hours per Day",
    min_value=0.0,
    max_value=12.0,
    step=0.25,
    key="slider_hours",
    on_change=sync_hours_slider,
    help="Select average study hours per day (0.0 to 12.0)."
)
st.sidebar.number_input(
    "Exact Hours",
    min_value=0.0,
    max_value=12.0,
    step=0.1,
    key="num_hours",
    on_change=sync_hours_num
)

# Feature 2: Sleep Hours
st.sidebar.subheader("2. 😴 Sleep Hours")
st.sidebar.slider(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=12.0,
    step=0.25,
    key="slider_sleep",
    on_change=sync_sleep_slider,
    help="Select average sleep hours per day (0.0 to 12.0)."
)
st.sidebar.number_input(
    "Exact Sleep",
    min_value=0.0,
    max_value=12.0,
    step=0.1,
    key="num_sleep",
    on_change=sync_sleep_num
)

# Feature 3: Attendance Percentage
st.sidebar.subheader("3. 🏫 Attendance Percentage")
st.sidebar.slider(
    "Attendance %",
    min_value=0.0,
    max_value=100.0,
    step=1.0,
    key="slider_att",
    on_change=sync_att_slider,
    help="Select attendance percentage (0 to 100%)."
)
st.sidebar.number_input(
    "Exact Attendance %",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key="num_att",
    on_change=sync_att_num
)

# Feature 4: Previous Marks
st.sidebar.subheader("4. 📝 Previous Exam Marks")
st.sidebar.slider(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    step=1.0,
    key="slider_prev",
    on_change=sync_prev_slider,
    help="Select previous exam score (0 to 100)."
)
st.sidebar.number_input(
    "Exact Previous Marks",
    min_value=0.0,
    max_value=100.0,
    step=0.5,
    key="num_prev",
    on_change=sync_prev_num
)

# Read Final Input Values from session state
f_hours = float(st.session_state.hours_studied)
f_sleep = float(st.session_state.sleep_hours)
f_att = float(st.session_state.attendance)
f_prev = float(st.session_state.previous_marks)

# Prediction Calculation
user_input_df = pd.DataFrame([[f_hours, f_sleep, f_att, f_prev]], columns=FEATURES)
raw_pred = float(multi_model.predict(user_input_df)[0])
clamped_score = max(0.0, min(100.0, raw_pred))
grade = get_performance_grade(clamped_score)

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📚 Multiple Linear Regression Student Predictor")
    st.markdown("Predict student exam performance based on **Study Hours**, **Sleep Hours**, **Attendance %**, and **Previous Marks**.")
with col_h2:
    st.markdown("<div style='text-align: right; margin-top: 15px;'><span class='badge-sample'>📌 Sample Dataset (100 Rows)</span></div>", unsafe_allow_html=True)

st.markdown("---")

# Prediction Hero Section
col_pred, col_formula = st.columns([1.2, 1])

with col_pred:
    st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 1.1rem; opacity: 0.9;">Predicted Score for Selected Features</div>
            <div class="prediction-score">{clamped_score:.2f}%</div>
            <div class="grade-badge">Grade: {grade}</div>
            <div style="font-size: 0.85rem; margin-top: 12px; opacity: 0.8;">
                Hours: {f_hours:.1f}h | Sleep: {f_sleep:.1f}h | Attendance: {f_att:.0f}% | Prev Marks: {f_prev:.0f}
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_formula:
    st.subheader("🧮 Multiple Linear Regression Formula")
    st.latex(r"\text{Marks} = (w_1 \cdot H) + (w_2 \cdot S) + (w_3 \cdot A) + (w_4 \cdot P) + c")
    
    cap_note = " \\text{ (capped at 100\\%)}" if raw_pred > 100 else (" \\text{ (floored at 0\\%)}" if raw_pred < 0 else "")
    c_h, c_s, c_a, c_p = raw_coefs
    
    note = " (capped at 100%)" if raw_pred > 100 else (" (floored at 0%)" if raw_pred < 0 else "")
    st.markdown("**Formula Breakdown:**")
    st.code(
        f"Score = ({c_h:.4f} x {f_hours:.2f}) + ({c_s:.4f} x {f_sleep:.2f})\n"
        f"      + ({c_a:.4f} x {f_att:.2f}) + ({c_p:.4f} x {f_prev:.2f})\n"
        f"      + ({intercept:.4f})\n"
        f"      = {raw_pred:.2f}{note}"
    )

# Model Comparison Section
st.markdown("---")
st.subheader("📊 Model Comparison: Simple vs Multiple Linear Regression")

comp_col1, comp_col2, comp_col3 = st.columns(3)

r2_diff = (r2_multi - r2_simple) * 100
mae_diff = mae_simple - mae_multi
rmse_diff = rmse_simple - rmse_multi

with comp_col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Variance Score</div>
            <div class="metric-value">{r2_multi:.4f} <span style="font-size: 1rem; color: #10b981;">({r2_multi*100:.1f}%)</span></div>
            <div style="font-size: 0.8rem; color: #34d399; margin-top: 4px;">▲ +{r2_diff:.1f}% vs 1-Feature ({r2_simple*100:.1f}%)</div>
        </div>
    """, unsafe_allow_html=True)

with comp_col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Absolute Error (MAE)</div>
            <div class="metric-value">{mae_multi:.2f} <span style="font-size: 1rem; color: #10b981;">marks</span></div>
            <div style="font-size: 0.8rem; color: #34d399; margin-top: 4px;">▼ -{mae_diff:.2f} marks vs 1-Feature ({mae_simple:.2f})</div>
        </div>
    """, unsafe_allow_html=True)

with comp_col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Root Mean Sq Error (RMSE)</div>
            <div class="metric-value">{rmse_multi:.2f} <span style="font-size: 1rem; color: #10b981;">marks</span></div>
            <div style="font-size: 0.8rem; color: #34d399; margin-top: 4px;">▼ -{rmse_diff:.2f} marks vs 1-Feature ({rmse_simple:.2f})</div>
        </div>
    """, unsafe_allow_html=True)

# Feature Importance & Impact Analysis
st.markdown("---")
st.subheader("🏆 Feature Importance & Impact Analysis")
st.markdown(f"Which feature affects student marks the most? Standardized beta coefficient analysis shows that **{top_feature_row['Feature']}** has the highest relative impact (**{top_feature_row['Impact %']:.1f}%**).")

feat_col1, feat_col2 = st.columns([1.2, 1])

with feat_col1:
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    fig_imp.patch.set_facecolor('#0f172a')
    ax_imp.set_facecolor('#1e293b')

    bars = ax_imp.barh(impact_df['Feature'], impact_df['Impact %'], color=['#38bdf8', '#34d399', '#818cf8', '#fbbf24'], edgecolor='#334155')
    ax_imp.set_title("Relative Feature Impact on Exam Score (%)", fontsize=13, color='#f8fafc', pad=12)
    ax_imp.set_xlabel("Impact Percentage (%)", fontsize=11, color='#94a3b8')
    ax_imp.tick_params(colors='#f8fafc')
    ax_imp.grid(True, color='#334155', linestyle='--', alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax_imp.text(w + 1.0, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', ha='left', va='center', fontweight='bold', color='#f8fafc')

    ax_imp.invert_yaxis()
    ax_imp.set_xlim(0, max(impact_df['Impact %']) + 12)
    st.pyplot(fig_imp)

with feat_col2:
    st.write("### 📋 Feature Weight & Impact Breakdown")
    display_impact = impact_df[['Feature', 'Raw Coef (w)', 'Standardized Beta', 'Impact %']].copy()
    display_impact['Impact %'] = display_impact['Impact %'].map('{:.1f}%'.format)
    display_impact['Raw Coef (w)'] = display_impact['Raw Coef (w)'].map('{:.4f}'.format)
    display_impact['Standardized Beta'] = display_impact['Standardized Beta'].map('{:.4f}'.format)
    st.dataframe(display_impact, use_container_width=True)

# Tabs Section for EDA, Actual vs Predicted & Dataset View
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📉 Model Fit & Correlation Heatmap", "🔍 Exploratory Data Analysis (EDA)", "📋 Sample Dataset (100 Rows)"])

with tab1:
    fit_col1, fit_col2 = st.columns(2)
    with fit_col1:
        st.write("### 🎯 Actual vs Predicted Marks")
        fig_fit, ax_fit = plt.subplots(figsize=(6, 4.5))
        fig_fit.patch.set_facecolor('#0f172a')
        ax_fit.set_facecolor('#1e293b')

        ax_fit.scatter(y_test, y_pred_multi, color='#818cf8', s=70, edgecolors='#4338ca', alpha=0.8, zorder=3, label='Test Predictions')
        ax_fit.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit (1:1)', linewidth=2)
        ax_fit.set_xlabel("Actual Marks", color='#94a3b8')
        ax_fit.set_ylabel("Predicted Marks", color='#94a3b8')
        ax_fit.tick_params(colors='#94a3b8')
        ax_fit.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax_fit.legend(facecolor='#1e293b', labelcolor='#f8fafc')
        st.pyplot(fig_fit)

    with fit_col2:
        st.write("### 🌡️ Feature Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
        fig_corr.patch.set_facecolor('#0f172a')
        ax_corr.set_facecolor('#1e293b')

        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues', ax=ax_corr, cbar=False)
        ax_corr.tick_params(colors='#94a3b8')
        st.pyplot(fig_corr)

with tab2:
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        st.write("### 📌 Null / Missing Values Check")
        null_df = pd.DataFrame(df.isnull().sum(), columns=['Null Count'])
        st.dataframe(null_df, use_container_width=True)
    with eda_col2:
        st.write("### 📈 Statistical Summary (100 Rows Sample)")
        st.dataframe(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']], use_container_width=True)

with tab3:
    st.write("### 📁 Student Sample Dataset (100 Rows)")
    st.caption("Realistic synthetic sample dataset created for model training & testing.")
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Multiple Linear Regression Project built with Python, Scikit-Learn, Pandas, NumPy, Matplotlib & Streamlit.")

