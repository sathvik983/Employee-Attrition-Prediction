import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def add_custom_features(df_input):
    """
    Computes custom domain features for attrition prediction.
    Can be run on the full training dataset or a single-row user input dataframe.
    """
    df = df_input.copy()
    
    # 1. Overall Satisfaction Score (Quality of Life)
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    # Check if all these columns exist in the input
    if all(col in df.columns for col in satisfaction_cols):
        df['SatisfactionSum'] = df[satisfaction_cols].sum(axis=1)
    else:
        df['SatisfactionSum'] = 10 # Neutral fallback
        
    # 2. Compensation relative to age
    if 'MonthlyIncome' in df.columns and 'Age' in df.columns:
        df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1e-5)
    else:
        df['IncomePerAge'] = 150.0
        
    # 3. Ratio of tenure at company to total working years
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        # Avoid division by zero
        df['YearsAtCompanyPerTotalYears'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1e-5)
    else:
        df['YearsAtCompanyPerTotalYears'] = 0.5

    # 4. Promotion Rate
    if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
        df['PromotionRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1e-5)
    else:
        df['PromotionRatio'] = 0.0

    # 5. Role stability ratio
    if 'YearsInCurrentRole' in df.columns and 'YearsAtCompany' in df.columns:
        df['YearsInCurrentRolePerYearsAtCompany'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1e-5)
    else:
        df['YearsInCurrentRolePerYearsAtCompany'] = 0.0
        
    return df

def get_preprocessor(numerical_features, categorical_features):
    """
    Creates a ColumnTransformer preprocessor for scaling and encoding.
    """
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ]
    )
    return preprocessor

def prepare_data(data_path="data/clean_attrition.csv", target_col='Attrition', test_size=0.2, random_state=42):
    """
    Loads data, engineers features, splits into train/test, and sets up the preprocessor.
    """
    df = pd.read_csv(data_path)
    
    # Engineer custom features
    df_engineered = add_custom_features(df)
    
    # Split features and target
    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]
    
    # Identify feature types
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)
    
    # Stratified split to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Set up preprocessor
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features = prepare_data()
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Fit preprocessor test
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    print(f"Transformed train features shape: {X_train_trans.shape}")
