import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, output:Path|None=None) -> None:
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        class_report = classification_report(y, y_pred,output_dict=True)
        print(class_report)
        if output is not None:
            class_report_df = pd.DataFrame(class_report).transpose()
            print(class_report_df)
            #class_report_df["Accuracy"] = acc 
            class_report_df.to_csv(output)
        #print(f"Accuracy: {acc:.4f}")
        #print("Classification Report:")
        #print(classification_report(y, y_pred))

        

if __name__ == "__main__":
    import src.data.preprocess as p
    from src.data.load_data import load_raw_migraine_data
    from src.config import DATE_COLUMN, PROCESSED_DATA_DIR
    df =  p.preprocess(load_raw_migraine_data())
    #df = p.fully_preprocess(load_raw_migraine_data(),options="lstm")
    df['days_since_start'] = (df[DATE_COLUMN] - df[DATE_COLUMN].min()).dt.days
    df = df.drop(columns=[DATE_COLUMN,"hydration_level"])
    
    X = df.drop(columns=['migraine_tomorrow', "migraine_occurrence"]) 
    y = df['migraine_tomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestModel(n_estimators=10000)
    rf_model.train(X_train, y_train)
    rf_model.evaluate(X_test, y_test,output=PROCESSED_DATA_DIR / "randomforest_report.csv")