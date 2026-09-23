# OncoPredict

## Machine Learning-Based Breast Cancer Classification and Data Analysis

OncoPredict is a Streamlit-based machine learning application designed to classify breast cancer cases using clinical cell-nuclei measurements. The project uses a trained Logistic Regression model with feature scaling to predict whether a tumor is **benign** or **malignant**.

The application also provides interactive visualizations and an additional data-analysis toolkit for exploring datasets and experimenting with machine learning models.

## Features

* Breast cancer classification using Machine Learning
* Logistic Regression prediction model
* Feature scaling using StandardScaler
* Prediction probability display
* Interactive input controls for cell-nuclei measurements
* Radar-chart visualization of input features
* Streamlit-based interactive user interface
* CSV data upload and exploration
* Data visualization and analysis
* Random Forest experimentation
* Downloadable and reusable trained model components

## Machine Learning Model

The main prediction system uses:

* **Algorithm:** Logistic Regression
* **Preprocessing:** StandardScaler
* **Task:** Binary classification
* **Classes:**

  * Benign
  * Malignant

The model uses cell-nuclei measurements as input features to generate a classification prediction and associated probability.

## Project Structure

```text
OncoPredict/
│
├── app/
│   ├── main.py
│   ├── pro.py
│   ├── model.pkl
│   ├── scaler.pkl
│   └── style.css
│
├── data/
│   └── breast-cancer.csv
│
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Plotly
* Matplotlib
* Seaborn

## How It Works

The application follows this general workflow:

```text
User Input
    ↓
Cell-Nuclei Measurements
    ↓
Feature Scaling
    ↓
Trained Logistic Regression Model
    ↓
Prediction
    ↓
Benign / Malignant
    ↓
Prediction Probability & Visualization
```

## Running the Application

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd OncoPredict
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
streamlit run app/main.py
```

The application will open in your browser.

## Disclaimer

This project is intended for **educational and demonstration purposes only**. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice or clinical diagnosis.


