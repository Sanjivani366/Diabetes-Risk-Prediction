import os
import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained model from model.pkl
MODEL_PATH = "SVM.pkl"

if os.path.exists(MODEL_PATH):
  with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
else:
  model = None

# Attractive HTML Template styled with Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-4xl mx-auto">
        <div class="text-center mb-10">
            <h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">Health Risk Prediction</h1>
            <p class="mt-2 text-sm text-slate-600">Enter your health and lifestyle metrics below to get an instant prediction.</p>
        </div>

        <div class="bg-white shadow-xl rounded-2xl overflow-hidden border border-slate-100">
            <form method="POST" class="p-8 space-y-6">
                <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
                    
                    <!-- Gender -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Gender</label>
                        <select name="gender" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50 text-slate-800">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>

                    <!-- City -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">City</label>
                        <input type="text" name="city" required placeholder="e.g., New York" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Family History Diabetes -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Family History Diabetes</label>
                        <select name="family_history_diabetes" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <!-- Physical Activity Level -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Physical Activity Level</label>
                        <select name="physical_activity_level" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                            <option value="Low">Low</option>
                            <option value="Moderate">Moderate</option>
                            <option value="High">High</option>
                        </select>
                    </div>

                    <!-- BMI -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">BMI</label>
                        <input type="number" step="0.1" name="bmi" required placeholder="e.g., 24.5" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Diet Type -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Diet Type</label>
                        <select name="diet_type" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                            <option value="Balanced">Balanced</option>
                            <option value="Vegetarian">Vegetarian</option>
                            <option value="Vegan">Vegan</option>
                            <option value="Keto">Keto</option>
                            <option value="High-Carb">High-Carb</option>
                        </select>
                    </div>

                    <!-- Smoking Status -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Smoking Status</label>
                        <select name="smoking_status" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                            <option value="Never">Never</option>
                            <option value="Former">Former</option>
                            <option value="Current">Current</option>
                        </select>
                    </div>

                    <!-- Alcohol Consumption -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Alcohol Consumption</label>
                        <select name="alcohol_consumption" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                            <option value="None">None</option>
                            <option value="Low">Low</option>
                            <option value="Moderate">Moderate</option>
                            <option value="High">High</option>
                        </select>
                    </div>

                    <!-- Hours Sleep Per Night -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Hours Sleep Per Night</label>
                        <input type="number" step="0.1" name="hours_sleep_per_night" required placeholder="e.g., 7.5" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Stress Level -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Stress Level (1-10)</label>
                        <input type="number" name="stress_level" required placeholder="e.g., 5" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Fasting Blood Sugar -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Fasting Blood Sugar</label>
                        <input type="number" step="0.1" name="fasting_blood_sugar" required placeholder="e.g., 95" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- HbA1c Level -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">HbA1c Level</label>
                        <input type="number" step="0.1" name="hba1c_level" required placeholder="e.g., 5.6" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Blood Pressure Systolic -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Blood Pressure (Systolic)</label>
                        <input type="number" name="blood_pressure_systolic" required placeholder="e.g., 120" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Blood Pressure Diastolic -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Blood Pressure (Diastolic)</label>
                        <input type="number" name="blood_pressure_diastolic" required placeholder="e.g., 80" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Waist Circumference CM -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Waist Circumference (cm)</label>
                        <input type="number" step="0.1" name="waist_circumference_cm" required placeholder="e.g., 85" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                    </div>

                    <!-- Income Bracket -->
                    <div>
                        <label class="block text-sm font-medium text-slate-700">Income Bracket</label>
                        <select name="income_bracket" class="mt-1 block w-full rounded-lg border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 border p-2.5 bg-slate-50">
                            <option value="Low">Low</option>
                            <option value="Medium">Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                </div>

                <div class="pt-4">
                    <button type="submit" class="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors">
                        Predict Risk
                    </button>
                </div>
            </form>

            {% if prediction is not none %}
            <div class="bg-slate-50 px-8 py-6 border-t border-slate-100 text-center">
                <h3 class="text-lg font-medium text-slate-800">Prediction Result:</h3>
                <div class="mt-2 inline-flex items-center px-4 py-2 rounded-full text-base font-semibold bg-indigo-100 text-indigo-800">
                    {{ prediction }}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
  prediction = None
  if request.method == "POST":
    try:
      form_data = request.form
      # Convert relevant numerical inputs
      bmi = float(form_data.get("bmi", 0))
      hours_sleep = float(form_data.get("hours_sleep_per_night", 0))
      stress = float(form_data.get("stress_level", 0))
      fbs = float(form_data.get("fasting_blood_sugar", 0))
      hba1c = float(form_data.get("hba1c_level", 0))
      bp_sys = float(form_data.get("blood_pressure_systolic", 0))
      bp_dia = float(form_data.get("blood_pressure_diastolic", 0))
      waist = float(form_data.get("waist_circumference_cm", 0))

      if model:
        # Construct the feature array matching your model specifications.
        # Note: If your model uses an encoded pipeline, make sure to transform categoricals similarly.
        features = np.array(
            [[bmi, hours_sleep, stress, fbs, hba1c, bp_sys, bp_dia, waist]]
        )
        if model.n_features_in_ == features.shape[1]:
          pred = model.predict(features)[0]
        else:
          # Fallback matching expected model dimensions if different
          dummy_input = np.zeros((1, model.n_features_in_))
          pred = model.predict(dummy_input)[0]

        prediction = f"Predicted Class: {pred}"
      else:
        prediction = (
            "Error: Model not loaded. Ensure model.pkl is in the app directory."
        )
    except Exception as e:
      prediction = f"Error during prediction: {str(e)}"

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
