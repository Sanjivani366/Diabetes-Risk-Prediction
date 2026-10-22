import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
with open('SVM.pkl', 'rb') as f:
    model = pickle.load(f)

# Attractive, modern HTML layout with Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen py-10 px-4">
    <div class="max-w-4xl mx-auto bg-white shadow-xl rounded-2xl overflow-hidden border border-slate-100">
        <div class="bg-gradient-to-r from-blue-600 to-indigo-700 py-6 px-8 text-white">
            <h1 class="text-2xl font-bold tracking-tight">Health & Lifestyle Risk Prediction</h1>
            <p class="text-blue-100 text-sm mt-1">Fill in the health parameters below to get instant model predictions.</p>
        </div>

        <form method="POST" class="p-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Gender -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Gender</label>
                <select name="gender" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <!-- City -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">City Code / Category</label>
                <input type="number" name="city" value="1" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Family History Diabetes -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Family History of Diabetes</label>
                <select name="family_history_diabetes" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <!-- Physical Activity Level -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Physical Activity Level (0-2)</label>
                <input type="number" step="0.1" name="physical_activity_level" value="1.0" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- BMI -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">BMI</label>
                <input type="number" step="0.1" name="bmi" value="24.5" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Diet Type -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Diet Type (Code)</label>
                <input type="number" name="diet_type" value="1" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Smoking Status -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Smoking Status (Code)</label>
                <input type="number" name="smoking_status" value="0" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Alcohol Consumption -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Alcohol Consumption (Code)</label>
                <input type="number" name="alcohol_consumption" value="0" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Hours Sleep Per Night -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Hours Sleep Per Night</label>
                <input type="number" step="0.1" name="hours_sleep_per_night" value="7.0" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Stress Level -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Stress Level (1-10)</label>
                <input type="number" name="stress_level" value="3" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Fasting Blood Sugar -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Fasting Blood Sugar</label>
                <input type="number" step="0.1" name="fasting_blood_sugar" value="95.0" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- HbA1c Level -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">HbA1c Level</label>
                <input type="number" step="0.1" name="hba1c_level" value="5.5" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Blood Pressure Systolic -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">BP Systolic</label>
                <input type="number" name="blood_pressure_systolic" value="120" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Blood Pressure Diastolic -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">BP Diastolic</label>
                <input type="number" name="blood_pressure_diastolic" value="80" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Waist Circumference CM -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Waist Circumference (cm)</label>
                <input type="number" step="0.1" name="waist_circumference_cm" value="85.0" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <!-- Income Bracket -->
            <div>
                <label class="block text-xs font-semibold uppercase text-slate-500 mb-1">Income Bracket</label>
                <input type="number" name="income_bracket" value="2" required class="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg shadow-md transition duration-200">
                    Predict Result
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="mx-8 mb-8 p-5 bg-slate-100 rounded-xl border border-slate-200 text-center">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-500">Prediction Result</h2>
            <p class="text-2xl font-bold text-blue-600 mt-1">
                {% if prediction == 1 %}
                    High Risk / Positive
                {% else %}
                    Low Risk / Negative
                {% endif %}
                <span class="text-xs text-slate-400 block font-normal">(Raw Output: {{ prediction }})</span>
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the exact order expected by the model
            features = [
                float(request.form['gender']),
                float(request.form['city']),
                float(request.form['family_history_diabetes']),
                float(request.form['physical_activity_level']),
                float(request.form['bmi']),
                float(request.form['diet_type']),
                float(request.form['smoking_status']),
                float(request.form['alcohol_consumption']),
                float(request.form['hours_sleep_per_night']),
                float(request.form['stress_level']),
                float(request.form['fasting_blood_sugar']),
                float(request.form['hba1c_level']),
                float(request.form['blood_pressure_systolic']),
                float(request.form['blood_pressure_diastolic']),
                float(request.form['waist_circumference_cm']),
                float(request.form['income_bracket'])
            ]
            
            # Reshape for single sample prediction
            final_features = np.array([features])
            prediction = model.predict(final_features)[0]
        except Exception as e:
            prediction = f"Error in calculation: {stre(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
