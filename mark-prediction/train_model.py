import os
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

FEATURES = ['Hours Studied', 'Sleep Hours', 'Attendance Percentage', 'Previous Marks']
TARGET = 'Marks Scored'

def run_pipeline():
    print("=" * 70)
    print("      STUDENT MARKS PREDICTOR - MULTIPLE LINEAR REGRESSION PIPELINE")
    print("=" * 70)

    # Directories setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # 1. Load Dataset
    data_path = os.path.join("data", "student_scores.csv")
    print(f"\n[1/7] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    print(f"\n---> Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n---> First 5 Rows of Dataset:")
    print(df.head())
    
    # 2. Exploratory Data Analysis (EDA)
    print("\n[2/7] Exploratory Data Analysis (EDA)...")
    print("\n---> Null Value Check:")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    print("\n---> Summary Statistics:")
    print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
    
    # Plot 1: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='Blues', vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title("Correlation Heatmap: Features vs Marks Scored", fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    plt.tight_layout()
    corr_path = os.path.join("static", "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"---> Correlation heatmap saved to '{corr_path}'")

    # Plot 2: Scatter plot of Hours Studied vs Marks Scored
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['Hours Studied'], df['Marks Scored'], color='#3b82f6', s=60, alpha=0.8, edgecolors='#1e40af', zorder=3)
    ax.set_title("Hours Studied vs Marks Scored (EDA)", fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    ax.set_xlabel("Hours Studied", fontsize=12, labelpad=10, color='#334155')
    ax.set_ylabel("Marks Scored", fontsize=12, labelpad=10, color='#334155')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    scatter_path = os.path.join("static", "eda_scatter_plot.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # 3. Train-Test Split
    print("\n[3/7] Splitting Dataset into Train and Test sets (80/20)...")
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set:  {len(X_test)} samples")
    
    # 4. Train Models (Simple vs Multiple Linear Regression)
    print("\n[4/7] Training Machine Learning Models...")
    
    # Model A: Simple Linear Regression (1 feature: Hours Studied)
    simple_model = LinearRegression()
    simple_model.fit(X_train[['Hours Studied']], y_train)
    y_pred_simple = simple_model.predict(X_test[['Hours Studied']])
    
    r2_simple = r2_score(y_test, y_pred_simple)
    mae_simple = mean_absolute_error(y_test, y_pred_simple)
    rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

    # Model B: Multiple Linear Regression (All 4 features)
    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_pred_multi = multi_model.predict(X_test)
    
    r2_multi = r2_score(y_test, y_pred_multi)
    mae_multi = mean_absolute_error(y_test, y_pred_multi)
    mse_multi = mean_squared_error(y_test, y_pred_multi)
    rmse_multi = np.sqrt(mse_multi)

    # 5. Model Comparison
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON: SIMPLE vs MULTIPLE LINEAR REGRESSION")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Simple LR (1 Feature)':<22} | {'Multiple LR (4 Features)':<22}")
    print("-" * 75)
    print(f"{'R2 Score (Variance)':<25} | {r2_simple:.4f} ({r2_simple*100:.2f}%)          | {r2_multi:.4f} ({r2_multi*100:.2f}%)")
    print(f"{'Mean Absolute Error (MAE)':<25} | {mae_simple:.4f}                 | {mae_multi:.4f}")
    print(f"{'Root Mean Sq Error (RMSE)':<25} | {rmse_simple:.4f}                 | {rmse_multi:.4f}")
    print("=" * 70)

    # 6. Feature Importance & Impact Analysis
    print("\n[5/7] Analyzing Feature Importance & Impact...")
    
    # Standardize features for normalized beta coefficient comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaled_model = LinearRegression()
    scaled_model.fit(X_train_scaled, y_train)
    
    raw_coefs = multi_model.coef_
    beta_coefs = scaled_model.coef_
    intercept = multi_model.intercept_
    
    impact_df = pd.DataFrame({
        'Feature': FEATURES,
        'Raw Coefficient (w)': raw_coefs,
        'Standardized Beta Coef': beta_coefs,
        'Abs Beta Impact': np.abs(beta_coefs)
    }).sort_values(by='Abs Beta Impact', ascending=False)
    
    total_impact = impact_df['Abs Beta Impact'].sum()
    impact_df['Impact Percentage (%)'] = (impact_df['Abs Beta Impact'] / total_impact) * 100

    print("\n---> Feature Importance Ranking (Which feature affects marks the most):")
    for idx, row in impact_df.iterrows():
        print(f"  Rank: {row['Feature']:<22} | Raw Coef: {row['Raw Coefficient (w)']:>7.4f} | Std Beta: {row['Standardized Beta Coef']:>7.4f} | Impact: {row['Impact Percentage (%)']:>5.1f}%")

    top_feature = impact_df.iloc[0]['Feature']
    top_impact = impact_df.iloc[0]['Impact Percentage (%)']
    print(f"\nMOST IMPACTFUL FEATURE: '{top_feature}' contributing {top_impact:.1f}% to the predicted exam score.")

    # Plot 3: Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(impact_df['Feature'], impact_df['Impact Percentage (%)'], color=['#3b82f6', '#10b981', '#6366f1', '#f59e0b'], edgecolor='#1e293b')
    ax.set_title("Feature Importance (% Impact on Final Marks)", fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    ax.set_xlabel("Impact Percentage (%)", fontsize=12, labelpad=10, color='#334155')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.8, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center', fontweight='bold', color='#1e293b')
        
    plt.gca().invert_yaxis()
    plt.tight_layout()
    importance_path = os.path.join("static", "feature_importance.png")
    plt.savefig(importance_path, dpi=300)
    plt.close()

    # Plot 4: Actual vs Predicted (Multiple LR)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred_multi, color='#8b5cf6', s=90, edgecolors='#5b21b6', zorder=3, label='Test Samples')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal 1:1 Line', linewidth=2)
    ax.set_title("Actual vs Predicted Marks (Multiple Linear Regression)", fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    ax.set_xlabel("Actual Marks", fontsize=12, labelpad=10, color='#334155')
    ax.set_ylabel("Predicted Marks", fontsize=12, labelpad=10, color='#334155')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # 7. Model Persistence
    print("\n[6/7] Saving Trained Model and Scaler...")
    joblib_model_path = os.path.join("models", "student_marks_model.joblib")
    pickle_model_path = os.path.join("models", "student_marks_model.pkl")
    
    model_payload = {
        'model': multi_model,
        'simple_model': simple_model,
        'scaler': scaler,
        'features': FEATURES,
        'raw_coefs': dict(zip(FEATURES, raw_coefs)),
        'intercept': float(intercept),
        'metrics': {
            'multi': {'r2': r2_multi, 'mae': mae_multi, 'rmse': rmse_multi, 'mse': mse_multi},
            'simple': {'r2': r2_simple, 'mae': mae_simple, 'rmse': rmse_simple}
        },
        'impact': impact_df.to_dict(orient='records')
    }
    
    joblib.dump(multi_model, joblib_model_path)
    joblib.dump(model_payload, os.path.join("models", "model_metadata.joblib"))
    with open(pickle_model_path, 'wb') as f:
        pickle.dump(multi_model, f)
        
    print(f"---> Multiple Linear Regression Model saved to '{joblib_model_path}'")
    
    # 8. Sample Prediction Test
    sample_input = pd.DataFrame([[8.5, 7.5, 90.0, 85.0]], columns=FEATURES)
    sample_pred = multi_model.predict(sample_input)[0]
    print("\n[7/7] Sample Prediction Test:")
    print(f"  Inputs: Hours=8.5, Sleep=7.5, Attendance=90%, PrevMarks=85")
    print(f"  Predicted Exam Score: {sample_pred:.2f} / 100")
    print("=" * 70)

    return model_payload

if __name__ == '__main__':
    run_pipeline()
