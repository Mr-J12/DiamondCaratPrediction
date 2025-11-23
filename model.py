import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

# --- App Title and Description ---
st.set_page_config(page_title='Crystalytics', page_icon='💎', layout='wide')
st.title("Diamond Carat Prediction")
st.write(
    "This application predicts the **carat** of a diamond based on its features. "
    "Use the sidebar to input the diamond's characteristics."
)

# --- Load and Cache the Data ---
@st.cache_data
def load_data():
    """Loads and preprocesses the diamond dataset."""
    try:
        df = pd.read_csv("diamonds.csv")
        if df.columns[0] == '':
            df = df.drop(df.columns[0], axis=1)
        return df
    except FileNotFoundError:
        st.error("Error: 'diamonds.csv' not found. Please ensure the file is in the correct directory.")
        return None

df = load_data()

if df is not None:
    # --- Sidebar for User Input ---
    st.sidebar.header("Input Diamond Features")

    # Create input fields for all features present in the original dataset
    cut_options = sorted(df['cut'].unique())
    color_options = sorted(df['color'].unique())
    clarity_options = sorted(df['clarity'].unique())

    # Categorical inputs
    cut = st.sidebar.selectbox("Cut", cut_options)
    color = st.sidebar.selectbox("Color", color_options)
    clarity = st.sidebar.selectbox("Clarity", clarity_options)

    # Numerical inputs with sliders
    depth = st.sidebar.slider("Depth", float(df['depth'].min()), float(df['depth'].max()), float(df['depth'].mean()))
    table = st.sidebar.slider("Table", float(df['table'].min()), float(df['table'].max()), float(df['table'].mean()))
    price = st.sidebar.slider("Price", float(df['price'].min()), float(df['price'].max()), float(df['price'].mean()))
    x = st.sidebar.slider("X (Length in mm)", float(df['x'].min()), float(df['x'].max()), float(df['x'].mean()))
    y = st.sidebar.slider("Y (Width in mm)", float(df['y'].min()), float(df['y'].max()), float(df['y'].mean()))
    z = st.sidebar.slider("Z (Depth in mm)", float(df['z'].min()), float(df['z'].max()), float(df['z'].mean()))

    # --- Train the Model (Cached) ---
    @st.cache_resource
    def train_model(df):
        """Preprocesses data and trains the Random Forest Regressor."""
        # One-Hot Encode categorical features
        df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)

        # Define features and target
        y = df_encoded['carat']
        # We drop 'price' as it's not a predictive feature for carat weight itself
        X = df_encoded.drop(['carat'], axis=1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Also return the columns for encoding the user input later
        return model, X.columns, X_train, X_test, y_train, y_test

    model, feature_columns, X_train, X_test, y_train, y_test = train_model(df)


    # --- Prediction Logic ---
    if st.sidebar.button("Predict Carat"):
        # Create a DataFrame from user inputs
        input_data = {
            'depth': [depth],
            'table': [table],
            'price': [price],
            'x': [x],
            'y': [y],
            'z': [z],
            'cut': [cut],
            'color': [color],
            'clarity': [clarity]
        }
        input_df = pd.DataFrame(input_data)

        # One-Hot Encode the user input DataFrame
        input_encoded = pd.get_dummies(input_df, columns=['cut', 'color', 'clarity'], drop_first=True)

        # Align columns with the training data, filling missing columns with 0
        input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_final)

        st.subheader("Prediction Result")
        st.metric(label="Predicted Carat Value", value=f"{prediction[0]:.2f}")

    # --- Display Model Performance ---
    st.subheader("Model Performance")
    st.write("The model used for prediction is a **Random Forest Regressor(RFR)**.")

    # Calculate and display R² score
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**R² Score on Test Data using RFR:** `{r2:.4f}`")

    # --- Display a sample of the data ---
    st.subheader("Sample of the Diamond Dataset")
    st.dataframe(df.head())
    
    st.subheader("Get a Random Sample")
    if st.button("Get Random Sample"):
        st.dataframe(df.sample(1))
