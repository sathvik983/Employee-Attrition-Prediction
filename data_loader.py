import os
import urllib.request
import pandas as pd

def download_and_clean_data():
    # Setup directories
    os.makedirs("data", exist_ok=True)
    raw_path = os.path.join("data", "raw_attrition.csv")
    clean_path = os.path.join("data", "clean_attrition.csv")
    
    # URL of dataset (verified in research phase)
    url = "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    # Download raw dataset if not present
    if not os.path.exists(raw_path):
        print(f"Downloading raw dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, raw_path)
            print("Download successful.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
    else:
        print("Raw dataset already exists locally.")

    # Load data
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset with shape: {df.shape}")

    # Columns to drop (either constant or identifier)
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    df_clean = df.drop(columns=existing_cols_to_drop)
    print(f"Dropped columns: {existing_cols_to_drop}")

    # Map target variable 'Attrition' to binary (1 = Yes, 0 = No)
    if 'Attrition' in df_clean.columns:
        unique_vals = df_clean['Attrition'].unique()
        if 'Yes' in unique_vals or 'No' in unique_vals:
            df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
            print("Mapped 'Attrition' values: 'Yes' -> 1, 'No' -> 0")

    # Map OverTime to binary or let categorical encoding handle it
    # We will let feature_engineering handle it, but OverTime is a key feature

    # Save cleaned dataset
    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to {clean_path}. Cleaned shape: {df_clean.shape}")
    
    return df_clean

if __name__ == "__main__":
    download_and_clean_data()
