#! /usr/bin/env python3
import joblib
import sys
import pandas as pd
import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------------------

print("-------------------------------------------------------------------------------------------------------------")
def encode_var(cat_data):
	# --------------------------------------------------------
	# Encode the categorical data as integers
	le = LabelEncoder()
	# Encode sex
	cat_data['sex'] = le.fit_transform(cat_data['sex'])
	# Encode genetic_diabetes
	cat_data['geneticDiabetes'] = le.fit_transform(cat_data['geneticDiabetes'])
	# Encode genetic_heart_disease
	cat_data['geneticHeartDiseases'] = le.fit_transform(cat_data['geneticHeartDiseases'])
	# Encode smoker
	cat_data['smoker'] = le.fit_transform(cat_data['smoker'])
	# Encode faint
	cat_data['faint'] = le.fit_transform(cat_data['faint'])
	# Encode sleep
	cat_data['sleep'] = le.fit_transform(cat_data['sleep'])


# -----------------------------------------------------------------------------------------------------------------

def predict_diagnosis(age, weight,height, sex,geneticDiabetes,geneticHeartDiseases,hr,hrv,sy_bp,dia_bp,rr,spo,temp,smoker,faint,sleep):
	# Load the model
	model = joblib.load("/home/ahmedshawki/CLOUD_PART/evaluation/model_RandomForest_fin.joblib")

	# Read the input data as JSON file
	
	json_string = f'{{"age": {age}, "weight": {weight}, "height": {height}, "sex": {sex}, "geneticDiabetes": {geneticDiabetes}, "geneticHeartDiseases": {geneticHeartDiseases}, "HR": {hr}, "systolic_BP": {sy_bp}, "diastolic_BP": {dia_bp}, "HRV": {hrv}, "SpO2": {spo}, "temperature": {temp},"RR": {rr},"smoker":{smoker}, "faint": {faint}, "sleep": {sleep}}}'

	input_data_dict = json.loads(json_string)
	input_data = pd.DataFrame.from_dict(input_data_dict, orient='index').T

	# Endode the input
	encode_var(input_data)

	# Convert input_data to a contiguous numpy array
	input_data = np.ascontiguousarray(input_data)

	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	# Make predictions
	predictions = model.predict(input_data)

	# decode
	predict_disease = {
		0 : "Angina" ,
		1 : "Arrhythmia" , 
		2 : "Asthma" ,
		3 : "COPD" ,
		4 : "Cardiac arrest" ,
		5 : "Cardiogenic shock" ,
		6 : "Healthy" ,
		7 : "Heart Attack" ,
		8 : "Hypertension" ,
		9 : "Pneumonia" 
	}

	# Convert the numpy array to a Python integer
	prediction = int(predictions.item())

	if prediction in predict_disease:
	    diagnosis = predict_disease[prediction]
	else:
	    diagnosis = "Unknown diagnosis"


	# Print the predictions to the console
	print(predictions)
	print(diagnosis)

	# Save the output to JSON file
	output_data_dict = {'diagnose': diagnosis}
	return output_data_dict
	#with open('output.json', 'w') as f:
	#    json.dump(output_data_dict, f)
	    
	    
#predict_diagnosis(55,60,159,0,0,0,90,100,119,80,12,99,37,0,0, 0)

