import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Import preprocessing from feature_engineering
from data_loader import download_and_clean_data
from feature_engineering import prepare_data

def train_and_evaluate_models():
    # Make sure data is ready
    if not os.path.exists("data/clean_attrition.csv"):
        download_and_clean_data()
        
    print("Preparing train and test splits...")
    X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features = prepare_data()
    
    # Define models and their parameter grids for GridSearchCV
    # We prefix hyperparameter names with 'classifier__' since they are inside a Pipeline
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'param_grid': {
                'classifier__max_depth': [3, 5, 8, 10],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 8, 12, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
        }
    }
    
    results = {}
    best_pipelines = {}
    
    for name, config in model_configs.items():
        print(f"\nTuning and training {name}...")
        
        # Create pipeline: Preprocessing followed by Classifier
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Grid search with 5-fold cross-validation, optimizing for F1-score due to class imbalance
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config['param_grid'],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on holdout test set
        best_model = grid_search.best_estimator_
        best_pipelines[name] = best_model
        
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Best Params for {name}: {grid_search.best_params_}")
        print(f"Test F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f}")
        
        results[name] = {
            'f1': float(f1),
            'roc_auc': float(auc),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'best_params': {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()},
            'confusion_matrix': cm.tolist()
        }
        
    # Select best model based on test F1-score
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\n==========================================")
    print(f"Selected Best Model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")
    print(f"==========================================")
    
    # Save the best pipeline
    os.makedirs("models", exist_ok=True)
    best_pipeline = best_pipelines[best_model_name]
    model_save_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model pipeline to {model_save_path}")
    
    # Save metadata results
    results_save_path = os.path.join("models", "training_results.json")
    with open(results_save_path, 'w') as f:
        json.dump({
            'best_model_name': best_model_name,
            'metrics': results
        }, f, indent=4)
    print(f"Saved training metrics to {results_save_path}")
    
    # Extract and save feature importance / coefficients for model interpretation
    extract_and_save_importances(best_pipeline, best_model_name, num_features, cat_features)

def extract_and_save_importances(pipeline, model_name, num_features, cat_features):
    """
    Extracts feature importances (for trees) or coefficients (for linear models)
    and saves them for dashboard visualization.
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Extract feature names from preprocessor
    cat_transformer = preprocessor.named_transformers_['cat']
    encoded_cat_features = list(cat_transformer.get_feature_names_out(cat_features))
    
    all_feature_names = num_features + encoded_cat_features
    
    importances = []
    
    if hasattr(classifier, 'feature_importances_'):
        # For Decision Tree and Random Forest
        importances = classifier.feature_importances_
        imp_type = 'importance'
    elif hasattr(classifier, 'coef_'):
        # For Logistic Regression
        importances = classifier.coef_[0]
        imp_type = 'coefficient'
    else:
        # Fallback
        importances = np.zeros(len(all_feature_names))
        imp_type = 'unknown'
        
    df_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Value': importances
    })
    
    # For linear coefficients, we sort by absolute value but keep sign for direction
    if imp_type == 'coefficient':
        df_imp['AbsValue'] = df_imp['Value'].abs()
        df_imp = df_imp.sort_values(by='AbsValue', ascending=False).drop(columns=['AbsValue'])
    else:
        df_imp = df_imp.sort_values(by='Value', ascending=False)
        
    importance_save_path = os.path.join("models", "feature_importance.csv")
    df_imp.to_csv(importance_save_path, index=False)
    print(f"Saved feature importances ({imp_type}) to {importance_save_path}")

if __name__ == "__main__":
    train_and_evaluate_models()
