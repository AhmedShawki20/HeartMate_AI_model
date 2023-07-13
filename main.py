#!/usr/bin/env python3

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import json
import os
from model_predict import predict_diagnosis

app = FastAPI()

class Patient(BaseModel):
    age: int
    weight: int
    height: float
    sex: int
    genetic_diabetes: int
    genetic_heart_disease: int
    HR: int
    HRV: int
    systolic_BP: int
    diastolic_BP: int
    RR: int
    SpO2: int
    temperature: float
    smoker: int
    faint: int
    sleep: int




@app.post("/predict")
async def send_data(patient: Patient):
    te = predict_diagnosis(patient.age, patient.weight, patient.height,patient.sex,patient.genetic_diabetes, patient.genetic_heart_disease,patient.HR,patient.HRV,patient.systolic_BP,patient.diastolic_BP,patient.RR,patient.SpO2,patient.temperature,patient.smoker,patient.faint,patient.sleep)
    return te


