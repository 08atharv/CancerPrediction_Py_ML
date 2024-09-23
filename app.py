# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
ds = pd.read_csv(r'./Datasets/The_Cancer_data_1500_V2.csv')
X = ds.drop(columns=['Diagnosis'])
y = ds['Diagnosis']

# Train/test split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    age = float(request.form['age'])
    gender = float(request.form['gender'])
    bmi = float(request.form['bmi'])
    smoking = float(request.form['smoking'])
    genetic_risk = float(request.form['genetic_risk'])
    physical_activity = float(request.form['physical_activity'])
    alcohol_intake = float(request.form['alcohol_intake'])
    cancer_history = float(request.form['cancer_history'])

    input_data = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'BMI': [bmi], 'Smoking': [smoking],
        'GeneticRisk': [genetic_risk], 'PhysicalActivity': [physical_activity],
        'AlcoholIntake': [alcohol_intake], 'CancerHistory': [cancer_history]
    })
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    result = "The patient is likely to have cancer." if prediction[0] == 1 else "The patient is unlikely to have cancer."
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
