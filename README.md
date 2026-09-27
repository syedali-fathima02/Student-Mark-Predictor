# Student Marks Predictor

A Python and Streamlit project that predicts a student's marks from study and other inputs. It compares a simple linear regression model (v1) with a multiple linear regression model (v2).

> **Note:** The models were trained on 100 rows of **sample (synthetic) student data**. The results below describe performance on this sample project. This is a learning project, not a reliable way to predict a real student's exam result.

## What the app shows

- Sliders to enter student details and see a prediction
- A breakdown of the prediction formula
- A feature-weight table with raw coefficients and standardized beta values
- An actual-versus-predicted chart
- A correlation heatmap

The v2 model uses hours studied, sleep hours, attendance, and previous marks. The charts and model results are based on the dataset, so moving the sliders changes the new prediction but does not retrain the model or move the existing data points.

## Model results

| Model | R² score | Mean absolute error (MAE) |
| --- | ---: | ---: |
| v1: simple linear regression | 71.6% | 6.13 marks |
| v2: multiple linear regression | 94.7% | 2.52 marks |

A higher R² and lower MAE indicate a better fit in this comparison. These scores come from the sample data and do not show how the model would perform on real students.

## Run locally

Use Python 3.11. Download or clone this project, open a terminal in the repository folder, and run:

```bash
cd mark-prediction
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project files

- `mark-prediction/app.py` - Streamlit app
- `mark-prediction/data/student_scores.csv` - synthetic sample dataset
- `mark-prediction/train_model.py` - training script
- `mark-prediction/requirements.txt` - Python packages

## Limitations

The dataset is synthetic, not collected from real students. A correlation or model coefficient does not prove that changing one input will cause a student's marks to change. Use this app to learn about regression and model evaluation, not to make academic decisions.
