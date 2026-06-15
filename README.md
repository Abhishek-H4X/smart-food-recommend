# smart-food-recommend
A Smart Food Recommendation System that uses Random Forest Classifiers to predict and suggest personalized breakfast, lunch, and dinner plans based on a user's physical attributes, activity levels, and health conditions. Powered by Python, Scikit-Learn, and Streamlit.



Markdown
# 🍎 Smart Food Recommendation System

An interactive web application built using **Streamlit** and **Scikit-Learn** that provides personalized daily meal recommendations (Breakfast, Lunch, and Dinner). The system analyzes a user's physical attributes, activity levels, and health conditions to predict the most suitable dietary plan using machine learning.

---

## 🚀 Key Features

* **Multi-Target Prediction:** Utilizes three separate Machine Learning models specialized in independently predicting optimal choices for breakfast, lunch, and dinner.
* **Interactive User Interface:** Features a clean, responsive web dashboard built with Streamlit, allowing users to effortlessly input data and view instant recommendations.
* **Robust Data Preprocessing:** Implements automatic text normalization (stripping whitespaces and converting case) to ensure data consistency between user inputs and training data.
* **Safe Encoding Pipeline:** Features a customized fallback mechanism for categorical encoding, preventing application crashes if a user inputs an unseen category.

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Backend:** Scikit-Learn (Random Forest Classifier, Label Encoder)
* **Web Framework:** Streamlit
* **Model Serialization:** Pickle (for saving and loading trained models and encoders)

---

## 📁 Project Architecture

```text
├── models/                     # Directory storing serialized models and encoders (.pkl)
│   ├── breakfast.pkl
│   ├── lunch.pkl
│   ├── dinner.pkl
│   └── [gender/activity/disease/bmi].pkl
├── app.py                      # Streamlit frontend application code
├── model.py                    # Machine Learning training pipeline script
└── smart_food_dataset.csv      # Source CSV dataset used for training
⚙️ Setup and Installation Guide
Follow these step-by-step instructions to set up and run the project locally on your machine.

1. Clone the Repository
Clone this repository to your local system and navigate into the project directory:

Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
2. Install Dependencies
Ensure you have Python installed, then install the required libraries using pip:

Bash
pip install streamlit pandas numpy scikit-learn
3. Train the Machine Learning Models
Before running the web application, execute the training script. This script processes the raw dataset, trains three Random Forest Classifiers, and saves the models into the models/ directory:

Bash
python model.py
4. Launch the Application
Start the Streamlit local web server to interact with the system user interface:

Bash
streamlit run app.py
🧠 How It Works
Data Cleaning: The system drops metadata columns (like Name), handles missing health strings by converting empty values to 'none', and normalizes all categorical text to lowercase.

Feature Engineering: Features like Gender, Activity Level, Disease, and BMI Category are transformed from text to numerical format using Scikit-Learn's LabelEncoder.

Model Training: Three independent RandomForestClassifier models (configured with 100 estimators each) learn the patterns mapping physical profiles to specific meal recommendations.

Deployment & Inference: The Streamlit app loads these static files, securely maps real-time UI selections back into their encoded integers, and displays the corresponding predicted meal menu items.
