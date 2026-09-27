import os
import sys
import joblib
import pandas as pd

FEATURES = ['Hours Studied', 'Sleep Hours', 'Attendance Percentage', 'Previous Marks']

def get_performance_grade(score):
    if score >= 90:
        return "A+ (Outstanding)", "#10b981"
    elif score >= 80:
        return "A (Excellent)", "#3b82f6"
    elif score >= 70:
        return "B (Good)", "#6366f1"
    elif score >= 60:
        return "C (Satisfactory)", "#f59e0b"
    elif score >= 50:
        return "D (Pass)", "#eab308"
    else:
        return "F (Needs Improvement)", "#ef4444"

def predict_score(hours, sleep, attendance, prev_marks, model_path=os.path.join("models", "student_marks_model.joblib")):
    if not os.path.exists(model_path):
        print(f"Error: Trained model file not found at '{model_path}'.")
        print("Please run 'python train_model.py' first to train and save the model.")
        return None

    model = joblib.load(model_path)
    
    input_df = pd.DataFrame([[hours, sleep, attendance, prev_marks]], columns=FEATURES)
    raw_pred = float(model.predict(input_df)[0])
    clamped_score = max(0.0, min(100.0, raw_pred))
    
    coefs = dict(zip(FEATURES, model.coef_))
    intercept = float(model.intercept_)
    grade, color = get_performance_grade(clamped_score)

    return {
        "hours": hours,
        "sleep": sleep,
        "attendance": attendance,
        "prev_marks": prev_marks,
        "raw_prediction": raw_pred,
        "clamped_score": clamped_score,
        "coefs": coefs,
        "intercept": intercept,
        "grade": grade
    }

def main():
    print("=" * 65)
    print("      STUDENT MARKS PREDICTOR - MULTIPLE LINEAR REGRESSION CLI")
    print("=" * 65)

    model_path = os.path.join("models", "student_marks_model.joblib")
    if not os.path.exists(model_path):
        print("\n[!] Model not found. Training model now...")
        from train_model import run_pipeline
        run_pipeline()
    
    # CLI Arguments check (hours sleep attendance prev_marks)
    if len(sys.argv) >= 5:
        try:
            h = float(sys.argv[1])
            s = float(sys.argv[2])
            a = float(sys.argv[3])
            p = float(sys.argv[4])
            res = predict_score(h, s, a, p, model_path)
            print_result(res)
            return
        except ValueError:
            print("Invalid numeric arguments provided.")
            sys.exit(1)
            
    # Interactive prompt mode
    print("\nEnter input parameters for student score prediction:")
    try:
        h = float(input("  1. Hours Studied per Day (0-12): ").strip() or 6.5)
        s = float(input("  2. Sleep Hours per Day (0-12): ").strip() or 7.0)
        a = float(input("  3. Attendance Percentage (0-100%): ").strip() or 80.0)
        p = float(input("  4. Previous Exam Marks (0-100): ").strip() or 70.0)
        
        res = predict_score(h, s, a, p, model_path)
        print_result(res)

    except (ValueError, KeyboardInterrupt):
        print("\nExiting Student Marks Predictor.")

def print_result(res):
    print("-" * 65)
    print(f" INPUT FEATURES:")
    print(f"   • Hours Studied:         {res['hours']:.2f} hrs/day")
    print(f"   • Sleep Hours:           {res['sleep']:.2f} hrs/day")
    print(f"   • Attendance Percentage: {res['attendance']:.1f}%")
    print(f"   • Previous Marks:        {res['prev_marks']:.1f}")
    print(f"\n PREDICTED EXAM SCORE:     {res['clamped_score']:.2f} / 100")
    print(f" ESTIMATED GRADE:          {res['grade']}")
    
    c = res['coefs']
    print(f"\n FORMULA BREAKDOWN:")
    print(f"   Score = ({c['Hours Studied']:.4f} * {res['hours']}) + ({c['Sleep Hours']:.4f} * {res['sleep']}) + ({c['Attendance Percentage']:.4f} * {res['attendance']}) + ({c['Previous Marks']:.4f} * {res['prev_marks']}) + {res['intercept']:.4f}")
    print(f"   Raw Result: {res['raw_prediction']:.2f}")
    print("-" * 65)

if __name__ == '__main__':
    main()
