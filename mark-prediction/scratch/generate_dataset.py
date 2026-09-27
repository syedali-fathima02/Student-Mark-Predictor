import os
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500

# Generating realistic features
hours_studied = np.round(np.random.uniform(1.0, 10.0, n_samples), 1)
sleep_hours = np.round(np.random.uniform(4.0, 9.5, n_samples), 1)
attendance = np.round(np.random.uniform(55.0, 100.0, n_samples), 1)
previous_marks = np.round(np.random.uniform(40.0, 98.0, n_samples), 1)

# Synthetic equation with slight random noise
noise = np.random.normal(0, 3.2, n_samples)
marks_scored = (
    4.25 * hours_studied +
    1.15 * sleep_hours +
    0.28 * attendance +
    0.32 * previous_marks -
    18.5 +
    noise
)

# Clamp marks scored between 10 and 100
marks_scored = np.round(np.clip(marks_scored, 10.0, 100.0), 1)

df = pd.DataFrame({
    'Hours Studied': hours_studied,
    'Sleep Hours': sleep_hours,
    'Attendance Percentage': attendance,
    'Previous Marks': previous_marks,
    'Marks Scored': marks_scored
})

os.makedirs('data', exist_ok=True)
data_path = os.path.join('data', 'student_scores.csv')
df.to_csv(data_path, index=False)

print(f"Generated sample dataset with {len(df)} rows at '{data_path}'.")
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
