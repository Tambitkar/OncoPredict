import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label, min_value=float(0), max_value=float(data[key].max()), value=float(data[key].mean())
        )
    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = [
        'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 
        'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension'
    ]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    return fig


def add_predictions(input_data):
    model = pickle.load(open("path/to/model.pkl", "rb"))
    scaler = pickle.load(open("path/to/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def advanced_data_science_toolkit():
    st.title("Advanced Data Science Toolkit")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        # Numerical Processing with NumPy
        st.subheader("Numerical Processing with NumPy")
        arr_1d = np.array([1, 2, 3, 4, 5])
        st.write("1D Array:", arr_1d)
        st.write("Data Type of the Array:", arr_1d.dtype)
        arr_2d = arr_1d.reshape((1, -1))
        st.write("Reshaped 1D to 2D Array:", arr_2d)
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        stacked_arr = np.vstack((arr1, arr2))
        st.write("Stacked Arrays (Vertical):", stacked_arr)
        unstacked_arr1, unstacked_arr2 = np.vsplit(stacked_arr, 2)
        st.write("Unstacked Arrays:", unstacked_arr1, unstacked_arr2)
        st.subheader("Data Wrangling with Pandas")
        if st.checkbox("Show Non-Missing Data"):
            st.write(df.notnull())
        if st.checkbox("Drop Missing Data"):
            df = df.dropna()
            st.success("Dropped missing data.")
        if st.checkbox("Replace Missing Data"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            if not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            if not non_numeric_cols.empty:
                df[non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")
            st.success("Replaced missing data.")
        st.subheader("Exploratory Data Analysis with Pandas")
        if st.checkbox("Show Descriptive Statistics"):
            st.write(df.describe())
        if st.checkbox("Show Value Counts"):
            col = st.selectbox("Select Column for Value Counts", df.columns)
            st.write(df[col].value_counts())
        if st.checkbox("Show Transpose"):
            st.write(df.T)
        if st.checkbox("Show Aggregates"):
            agg_func = st.selectbox("Select Aggregate Function", ["sum", "mean", "min", "max"])
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                st.error("No numeric columns available for aggregation.")
            else:
                if agg_func == "sum":
                    st.write(numeric_df.sum())
                elif agg_func == "mean":
                    st.write(numeric_df.mean())
                elif agg_func == "min":
                    st.write(numeric_df.min())
                elif agg_func == "max":
                    st.write(numeric_df.max())
        st.subheader("Data Visualization")
        if st.checkbox("Show Line Plot"):
            selected_col = st.selectbox("Select a column for line plot", df.columns)
            st.line_chart(df[selected_col].dropna())
        if st.checkbox("Show Histogram"):
            selected_col = st.selectbox("Select a column for histogram", df.columns)
            plt.figure(figsize=(10, 5))
            sns.histplot(df[selected_col].dropna(), kde=True)
            st.pyplot(plt)
        st.subheader("Machine Learning")
        target = st.selectbox("Select Target Column", df.columns)
        feature_cols = st.multiselect("Select Feature Columns", df.columns)
        if target and feature_cols:
            X = df[feature_cols]
            y = df[target]
            test_size = st.slider("Select Test Size", 0.1, 0.9, 0.3)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model_type = st.selectbox("Select Model Type", ["RandomForestClassifier", "RandomForestRegressor"])
            if model_type == "RandomForestClassifier":
                model = RandomForestClassifier()
            elif model_type == "RandomForestRegressor":
                model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if model_type == "RandomForestClassifier":
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy}")
            elif model_type == "RandomForestRegressor":
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse}")


def main():
    st.title("Integrated Streamlit Application")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Breast Cancer Prediction", "Advanced Data Science Toolkit"])

    if app_mode == "Breast Cancer Prediction":
        input_data = add_sidebar()
        fig = get_radar_chart(input_data)
        st.plotly_chart(fig)
        add_predictions(input_data)
    elif app_mode == "Advanced Data Science Toolkit":
        advanced_data_science_toolkit()

if __name__ == "__main__":
    main()
