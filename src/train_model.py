from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_raw_migraine_data 
from src.data import preprocess
from src.models.lstm_model import LSTMModel 
import src.features.lstm_features as ftrs
from src.config import RAW_DATA_DIR,DATA_DIR, TRUE_VARS, MODELS_DIR,USER_ID,USER_ID_COLUMN, ADDED_COLS
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def train_lstm_model():
    df = load_raw_migraine_data(RAW_DATA_DIR / "migraine_dataset.csv")

    df = preprocess.preprocess(df)
    df.to_csv(DATA_DIR / "processed/clean_data.csv")

    #feature creation
    df_cyclical = ftrs.create_cyclical_feature(df)
    df_cyclical.to_csv(DATA_DIR / "processed/clean_data_featured.csv")
    #truncate for LSTM 
    df = preprocess.truncate(df_cyclical)
    df.to_csv(DATA_DIR / "processed/clean_data_featured_truncated.csv")

    #training part
    time_steps = 23
    feature_columns = (TRUE_VARS + ADDED_COLS)
    X_data, y_data, scaler = ftrs.create_sequences(df,time_steps=time_steps,feature_columns=feature_columns)

    split_index = int(0.80 * len(X_data))
    X_train, X_val = X_data[:split_index], X_data[split_index:]
    y_train, y_val = y_data[:split_index], y_data[split_index:]

    lstm_model = LSTMModel(time_steps=time_steps,n_features=len(feature_columns))
    lstm_model.build_model()
    history = lstm_model.train(X_train,y_train,validation_data=(X_val,y_val))

    y_val_pred = lstm_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    class_report = classification_report(y_val, y_val_pred,output_dict=True)
    cm = confusion_matrix(y_val, y_val_pred)
    

    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(DATA_DIR / "processed" / "lstm_report.csv", index=True)

    print("Accuracy:", acc)

    plot_training_history(history)

    lstm_model.save_model(MODELS_DIR / "trained_models" / "lstm1.keras")
    joblib.dump(scaler,MODELS_DIR / "trained_models"/ "scaler.pkl")

    plot_predictions_for_user(df,lstm_model,user_id=5,time_steps=time_steps,scaler=scaler, feature_cols=feature_columns)

def plot_training_history(history):
    """ Plot training history for loss and accuracy """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title('Model Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Model Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()

def plot_predictions_for_user(df, lstm_model, user_id, time_steps, scaler, feature_cols):
    """ Plot predictions vs reality for one specific user """
    user_data = df[df[USER_ID_COLUMN] == user_id]

    X_user, y_user, _ = ftrs.create_sequences(user_data, time_steps,feature_columns=feature_cols)

    X_user_scaled = scaler.transform(X_user.reshape(-1, X_user.shape[-1])).reshape(X_user.shape)
    predictions = lstm_model.predict(X_user_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(user_data['date'], user_data['migraine_tomorrow'], label='Actual', color='blue')
    ax.plot(user_data['date'].iloc[time_steps:], predictions, label='Predicted', color='red')

    ax.set_xlabel('Date')
    ax.set_ylabel('Migraine Occurrence')
    ax.set_title(f'Migraine Prediction vs Reality for User {user_id}')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    train_lstm_model()




