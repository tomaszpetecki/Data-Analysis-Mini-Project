from pathlib import Path
import pandas as pd

from src.config import RAW_MIGRAINE_CSV, USER_ID_COLUMN

def load_raw_migraine_data(path: Path|None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else RAW_MIGRAINE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Wrong path for the raw data: {csv_path}")
    
    df = pd.read_csv(csv_path)

    return df

#Let's check the structure of our data set
if __name__ == "__main__":
    df = load_raw_migraine_data()
    df_first_n = df.groupby('user_id', group_keys=False).apply(lambda x: x.head(90)).reset_index(drop=True)
    print(df_first_n.groupby(USER_ID_COLUMN).size())
    print("Shape:",df.shape)
