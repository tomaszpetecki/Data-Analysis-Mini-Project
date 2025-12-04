import streamlit as st
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR,USER_ID,USER_ID_COLUMN,TRUE_VARS,MODELS_DIR,ADDED_COLS
import numpy as np
from src.data.preprocess import preprocess, fully_preprocess
from src.features.lstm_features import create_sequences,create_cyclical_feature
from keras.models import load_model
import joblib
import plotly.graph_objects as go

def load_trained_model_and_scaler():
    model = load_model(MODELS_DIR/"trained_models"/"lstm1.keras")
    scaler = joblib.load(MODELS_DIR/"trained_models"/"scaler.pkl")
    return model, scaler

# Function to make multi-day predictions (iteratively - naive)
def predict_multiple_days(model, X_input, n_days, scaler):
    predictions = []

    current_input = X_input[-1].reshape(1, X_input.shape[1], X_input.shape[2])  # Reshape to (1, time_steps, n_features)

    for _ in range(n_days):
        # Predict the next day
        pred = model.predict(current_input)
        predictions.append(pred[0][0]) 

        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, 0] = pred 

    return np.array(predictions)

def plot_predictions_for_user(df, predictions, n_days):
    """ Plot interactive bar chart of actual vs predicted migraine occurrences for one specific user """
    user_data = df

    actual_data = user_data[['date', 'migraine_occurrence']].copy()

    actual_bar = go.Bar(
        x=actual_data['date'],
        y=actual_data['migraine_occurrence'],
        name='Your data',
        marker=dict(color='blue')
    )
    last_day = pd.to_datetime(user_data["date"].iloc[-1])
    predicted_dates = pd.date_range(start=last_day+pd.Timedelta(days=1), end=last_day+pd.Timedelta(days=3))
    predicted_data = pd.DataFrame({
        'date': predicted_dates,
        'predicted': predictions
    })

    predicted_bar = go.Bar(
        x=predicted_data['date'],
        y=predicted_data['predicted'],
        name='Predicted probability',
        marker=dict(color='orange')
    )

    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Migraine Occurrence'),
        barmode='group',
        showlegend=True
    )

    fig = go.Figure(data=[actual_bar, predicted_bar], layout=layout)
    fig.update_xaxes(range=[last_day -pd.Timedelta(days=4),last_day+pd.Timedelta(days=3.5)])

    st.plotly_chart(fig)

def main():
    raw_user_data = pd.read_csv(RAW_DATA_DIR / "migraine_dataset.csv")
    user_data = fully_preprocess(raw_user_data, options="lstm")
    user_data = user_data[user_data[USER_ID_COLUMN]==USER_ID]
    user_data_for_display = preprocess(raw_user_data)
    user_data_for_display = user_data_for_display[user_data_for_display[USER_ID_COLUMN]==USER_ID].drop(columns=USER_ID_COLUMN).set_index("date")
    
    st.set_page_config(
        page_title="Migraine Prediction App",
        layout="wide"  
    )
    st.title("HeadsUP! - Migraine Prediction App")
    col1,col2 = st.columns(2)
    col1.write("""
    **Hello!** 👋 Welcome to the Migraine Prediction App, a tool designed to help you predict whether you'll experience a migraine 
    in the upcoming days based on your health data.

    **How it works:**
    - The app uses a trained machine learning model to analyze your health data and predict migraine occurrences.
    - You can either use the **default demo user data** to explore the predictions, or you can **upload your own CSV file** with your migraine and health data.
    - The app will show you a comparison of **actual migraine occurrences** (based on past data) and **predictions** for the next 3 days.

    **What you need to do:**
    1. Upload your **CSV file** (containing your health data and migraine occurrences).
    2. The app will process the data and display a **bar chart** comparing actual data and predicted migraine occurrences.

    **The goal:**
    - You will be able to see predictions for the **next 3 days** and evaluate how well the model predicts migraine occurrences based on your health data.

    **Let's get started!**
    """)
    uploaded_file = col2.file_uploader("Upload your own .csv health data",type=["csv"])
    
    if uploaded_file is not None:
        try:
            uploaded_data  = pd.read_csv(uploaded_file)
            if len([i for i in TRUE_VARS if i not in uploaded_data.columns]) != 0:
                col2.error("Columns are wrong")
            if "date" not in uploaded_data.columns.map(lambda x: x.lower().strip()):
                col2.error("Does not contain date")
            
            user_data = preprocess(uploaded_data)
            user_data_for_display= user_data
            user_data = fully_preprocess(user_data)

        except Exception:
            pass
    
    col2.write(user_data_for_display)

    model, scaler = load_trained_model_and_scaler()
    
    X_input,y_input, _ = create_sequences(user_data, time_steps=15, feature_columns=TRUE_VARS + ADDED_COLS)  # Adjust function if necessary
    X_input_scaled = scaler.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)

    predictions = predict_multiple_days(model, X_input_scaled, n_days=3, scaler=scaler)

    
    plot_predictions_for_user(user_data, predictions, n_days=3)

main()
    

