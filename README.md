## Customer Churn Prediction

An end-to-end Machine Learning classification project that predicts whether a customer is likely to churn (Yes/No).

The project covers the complete machine learning workflow, from data preprocessing and model training to saving the final pipeline, building an interactive Streamlit web application, uploading the project to GitHub, and deploying it using Streamlit Community Cloud.

🚀 Live Demo

🔗 Streamlit App: 

🔗 GitHub Repository: https://github.com/arqamowais/CustomerChurnPrediction-DSAI

📌 Project Goal

The goal of this project is to build a machine learning classification model that can predict customer churn based on customer-related features.

The final solution allows users to enter customer information through a web interface and receive a prediction:
| Value | Meaning |
|---|---|
| Yes | Customer churned |
| No | Customer did not churn |

## 🛠️ Technologies Used
- Python
- Pandas — Data manipulation and analysis
- NumPy — Numerical computing
- Scikit-learn — Machine learning and preprocessing
- Matplotlib & Seaborn — Data visualization
- Joblib & Pickle — Model serialization
- Streamlit — Web application
- Git & GitHub — Version control and project hosting
- Streamlit Community Cloud — Deployment

## 🔄 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Train-Test Split
- Model Training
- Model Evaluation
- Model/Pipeline Serialization
- Streamlit Application Development
- GitHub Upload
- Deployment on Streamlit Community Cloud
📊 Machine Learning Model

This is a binary classification problem where the target variable is customer churn.

Target Variable
Value	Meaning
Yes	Customer is likely to churn
No Customer is likely to stay

The preprocessing steps and trained machine learning model are combined into a single pipeline to ensure that the same transformations used during training are applied to new customer data.

The final trained pipeline is saved as a .pkl file and loaded by the Streamlit application.

## 📁 Project Structure
```
customer-churn-prediction/
│
├── customer_churn_data.csv
├── model_training.ipynb
├── churn_pipeline.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
## File Description
- customer_churn_data.csv — Contains the dataset used for training and evaluation.
- model_training.ipynb — Contains exploratory data analysis and model development.
- churn_pipeline.pkl — Contains the final trained ML pipeline.
- app.py — Streamlit web application.
- requirements.txt — Required Python packages.
- README.md — Project documentation.
- .gitignore — Files and folders excluded from GitHub.

💻 Run the Project Locally
1. Clone the Repository
git clone [https://github.com/your-username/customer-churn-prediction.git](https://github.com/arqamowais/CustomerChurnPrediction-DSAI.git)

2. Navigate to the Project Directory
cd customer-churn-prediction

3. Create a Conda Environment
conda create -n churn-env python=3.11


Activate the environment:

conda activate churn-env

4. Install Dependencies
pip install -r requirements.txt

5. Run the Streamlit Application
streamlit run app.py


The application will open in your browser.

## 🌐 Streamlit Web Application

The Streamlit application provides an interactive interface where users can enter customer information.

The application:

Accepts customer details from the user.
Processes the input using the saved preprocessing pipeline.
Sends the processed data to the trained classification model.
Predicts whether the customer is likely to churn.
Displays the prediction to the user.
Example
Customer Information
        ↓
Streamlit Input Form
        ↓
Preprocessing Pipeline
        ↓
Machine Learning Model
        ↓
Churn Prediction
        ↓
Yes / No

## 📈 Model Evaluation

The trained model is evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Example:

- Accuracy  : 76.25%
- Precision : 76.87%
- Recall    : 83.08%
- F1-Score  : 79.85%

Replace the values above with the actual results from your trained model.

## 📦 Model Serialization

The final trained machine learning pipeline is saved as a .pkl file.

Example:

import pickle

with open("churn_pipeline.pkl", "wb") as file:
    pickle.dump(churn_pipeline, file)


The Streamlit application loads the saved model:

with open("churn_pipeline.pkl", "rb") as file:
    loaded_pipeline = pickle.load(file)


Using a complete pipeline helps ensure consistent preprocessing between training and prediction.

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

Deployment steps:

Push the project to GitHub.
Log in to Streamlit Community Cloud.
Connect the GitHub repository.
Select app.py as the main application file.
Deploy the application.
Share the generated public URL.
## 🔮 Future Improvements

Possible improvements include:

Hyperparameter tuning
Testing additional classification algorithms
Improving model performance
Adding probability/confidence scores
Adding feature importance visualizations
Improving the Streamlit UI
Adding batch prediction using CSV uploads
Monitoring model performance after deployment
## 👨‍💻 Author

M. Arqam Owais

GitHub: https://github.com/arqamowais

⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub!
