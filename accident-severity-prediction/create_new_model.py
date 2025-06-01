import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def backup_old_model():
    """Backup the old model if it exists"""
    old_model_path = os.path.join(SCRIPT_DIR, 'trained_model.sav')
    backup_path = os.path.join(SCRIPT_DIR, 'trained_model_backup.sav')
    
    if os.path.exists(old_model_path):
        try:
            old_model = joblib.load(old_model_path)
            joblib.dump(old_model, backup_path)
            print("Old model backed up successfully as 'trained_model_backup.sav'")
            return True
        except Exception as e:
            print(f"Error backing up old model: {e}")
    else:
        print("No old model found to backup")
    return False

def train_new_model():
    """Train and save a new model with updated features"""
    try:
        # Use the correct path for the CSV file
        csv_path = os.path.join(SCRIPT_DIR, 'cleaned_accident_data_preprocessed (4).csv')
        print(f"Loading CSV file from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print("\nAccident Severity Distribution:")
        print(df['Accident_Severity'].value_counts())
        
        # Convert categorical variables to numerical
        le = LabelEncoder()
        categorical_columns = ['Accident_Severity', 'Light_Conditions', 'Road_Surface_Conditions', 
                             'Weather_Conditions', 'Urban_or_Rural_Area']
        
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col])
            if col == 'Accident_Severity':
                print("\nEncoded Accident Severity Mapping:")
                for i, label in enumerate(le.classes_):
                    print(f"{i}: {label}")
        
        # Select features and target (without Did_Police_Officer_Attend)
        features = ['Light_Conditions', 
                   'Road_Surface_Conditions', 'Speed_limit', 'Weather_Conditions',
                   'Latitude', 'Longitude']
        
        if not all(col in df.columns for col in features):
            print("Error: Some required columns not found in the CSV.")
            missing_cols = [col for col in features if col not in df.columns]
            print(f"Missing columns: {missing_cols}")
            return False
            
        X = df[features]
        y = df['Accident_Severity']
        
        print("\nFeatures used for training:")
        for feature in features:
            print(f"- {feature}")
        
        print("\nTraining new model...")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced',
            criterion='entropy'
        )
        
        model.fit(X, y)
        
        # Print feature importance
        print("\nFeature Importance:")
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance)
        
        # Save the new model in the script directory
        model_path = os.path.join(SCRIPT_DIR, 'trained_model_new.sav')
        joblib.dump(model, model_path)
        print(f"\nNew model saved successfully as '{model_path}'")
        return True
        
    except Exception as e:
        print(f"Error training new model: {e}")
        return False

if __name__ == "__main__":
    print("Starting model backup and retraining process...")
    print(f"Working directory: {SCRIPT_DIR}")
    
    # Backup old model
    backup_success = backup_old_model()
    
    # Train new model
    training_success = train_new_model()
    
    if backup_success and training_success:
        print("\nProcess completed successfully!")
        print("1. Old model backed up as 'trained_model_backup.sav'")
        print("2. New model created as 'trained_model_new.sav'")
    else:
        print("\nProcess completed with some issues:")
        if not backup_success:
            print("- Failed to backup old model")
        if not training_success:
            print("- Failed to train new model")
        sys.exit(1) 