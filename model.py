import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Create models folder
if not os.path.exists("models"):
    os.makedirs("models")

# Load dataset
data = pd.read_csv("smart_food_dataset.csv")

# Drop Name column
data = data.drop("Name", axis=1)

# Clean text columns
data['Gender'] = data['Gender'].astype(str).str.strip().str.lower()
data['Activity_Level'] = data['Activity_Level'].astype(str).str.strip().str.lower()
data['Disease'] = data['Disease'].astype(str).str.strip().str.lower()
data['BMI_Category'] = data['BMI_Category'].astype(str).str.strip().str.lower()

# Replace empty values
data['Disease'] = data['Disease'].replace('', 'none')

# Features
X = data[['Age','Weight','Height','Gender','Activity_Level','Disease','BMI_Category']]

# Targets
y_breakfast = data['Recommended_Breakfast']
y_lunch = data['Recommended_Lunch']
y_dinner = data['Recommended_Dinner']

# Label Encoding
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_disease = LabelEncoder()
le_bmi = LabelEncoder()

X['Gender'] = le_gender.fit_transform(X['Gender'])
X['Activity_Level'] = le_activity.fit_transform(X['Activity_Level'])
X['Disease'] = le_disease.fit_transform(X['Disease'])
X['BMI_Category'] = le_bmi.fit_transform(X['BMI_Category'])

print("Disease Classes:", le_disease.classes_)

# Train Models (Random Forest)
model_breakfast = RandomForestClassifier(n_estimators=100)
model_lunch = RandomForestClassifier(n_estimators=100)
model_dinner = RandomForestClassifier(n_estimators=100)

model_breakfast.fit(X, y_breakfast)
model_lunch.fit(X, y_lunch)
model_dinner.fit(X, y_dinner)

# Save models
pickle.dump(model_breakfast, open("models/breakfast.pkl", "wb"))
pickle.dump(model_lunch, open("models/lunch.pkl", "wb"))
pickle.dump(model_dinner, open("models/dinner.pkl", "wb"))

# Save encoders
pickle.dump(le_gender, open("models/gender.pkl", "wb"))
pickle.dump(le_activity, open("models/activity.pkl", "wb"))
pickle.dump(le_disease, open("models/disease.pkl", "wb"))
pickle.dump(le_bmi, open("models/bmi.pkl", "wb"))

print("✅ Random Forest Model Trained Successfully")