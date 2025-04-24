from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Added for static files
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import os
from typing import Dict

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
exam1_data = None
exam2_data = None
models: Dict[str, RandomForestRegressor] = {}
predicted_exam2 = None

SUBJECTS = ['physics', 'maths', 'english', 'chemistry', 'computer']

def clean_marks_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    for col in SUBJECTS:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df[col] = cleaned_df[col].mask(cleaned_df[col] < 0)
            cleaned_df[col] = cleaned_df[col].mask(cleaned_df[col] > 100, 100)
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            cleaned_df[col] = cleaned_df[col].fillna(0)
    return cleaned_df

@app.post("/upload_exam1/")
async def upload_exam1(file: UploadFile = File(...)):
    global exam1_data
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            exam1_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xls', '.xlsx')):
            exam1_data = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")

        exam1_data.columns = exam1_data.columns.str.replace('_marks', '').str.strip()
        required_columns = ['student_id'] + SUBJECTS
        if not all(col in exam1_data.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"Missing required columns. Needed: {required_columns}, Found: {list(exam1_data.columns)}")

        exam1_data = clean_marks_data(exam1_data)
        return {"message": "Exam1 data uploaded successfully", "records": len(exam1_data)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/train_models/")
async def train_models():
    global exam1_data, models
    if exam1_data is None:
        raise HTTPException(status_code=400, detail="Please upload Exam1 data first")
    try:
        for subject in SUBJECTS:
            features = [s for s in SUBJECTS if s != subject]
            X = exam1_data[features]
            y = exam1_data[subject]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            models[subject] = model
        return {"message": "Models trained successfully for all subjects"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@app.post("/predict_exam2/")
async def predict_exam2(file: UploadFile = File(...)):
    global exam1_data, models, predicted_exam2

    if exam1_data is None or not models:
        raise HTTPException(status_code=400, detail="Please upload Exam1 data and train models first")

    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            exam2_ids_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xls', '.xlsx')):
            exam2_ids_df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel")

        if 'student_id' not in exam2_ids_df.columns:
            raise HTTPException(status_code=400, detail="Missing 'student_id' column")

        predicted_data = exam2_ids_df.copy()
        matched_data = exam1_data[exam1_data['student_id'].isin(predicted_data['student_id'])].reset_index(drop=True)

        if matched_data.empty:
            raise HTTPException(status_code=400, detail="No matching student IDs found in Exam1 data")

        for subject in SUBJECTS:
            feature_subjects = [s for s in SUBJECTS if s != subject]
            model = models.get(subject)
            if model is None:
                raise HTTPException(status_code=500, detail=f"Model for {subject} not found")

            X_pred = matched_data[feature_subjects]
            predictions = model.predict(X_pred)
            predicted_data[f"predicted_{subject}"] = predictions

        predicted_exam2 = predicted_data
        return {"message": "Predictions generated successfully", "records": len(predicted_exam2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/student_marks_chart/{student_id}")
async def get_student_chart(student_id: int):
    global exam1_data, predicted_exam2

    if exam1_data is None or predicted_exam2 is None:
        raise HTTPException(status_code=400, detail="Please upload and predict data first")

    try:
        exam1_student = exam1_data[exam1_data['student_id'] == student_id]
        exam2_student = predicted_exam2[predicted_exam2['student_id'] == student_id]

        if len(exam1_student) == 0 or len(exam2_student) == 0:
            raise HTTPException(status_code=404, detail="Student ID not found")

        exam1_marks = [exam1_student[subject].values[0] for subject in SUBJECTS]
        exam2_marks = [exam2_student[f"predicted_{subject}"].values[0] for subject in SUBJECTS]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.pie(exam1_marks, labels=SUBJECTS, autopct='%1.1f%%')
        ax1.set_title(f'Exam1 Marks Distribution\nStudent {student_id}')

        ax2.pie(exam2_marks, labels=SUBJECTS, autopct='%1.1f%%')
        ax2.set_title(f'Predicted Exam2 Marks Distribution\nStudent {student_id}')

        chart_filename = f"student_{student_id}_marks_chart.png"
        plt.tight_layout()
        plt.savefig(chart_filename, bbox_inches='tight')
        plt.close()

        return FileResponse(chart_filename, media_type='image/png', filename=chart_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

# Serve frontend files (this replaces your original root route)
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")