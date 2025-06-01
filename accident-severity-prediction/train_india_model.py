import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from imblearn.over_sampling import RandomOverSampler

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load your data using the correct path
df = pd.read_csv(os.path.join(script_dir, 'cleaned_accident_data_with_location.csv'))
print("Total rows in dataset:", len(df))
print("Available columns:", df.columns.tolist())

# Print coordinate ranges in the data
print("\nCoordinate ranges in data:")
print("Latitude range:", df['Latitude'].min(), "to", df['Latitude'].max())
print("Longitude range:", df['Longitude'].min(), "to", df['Longitude'].max())

# Filter for Indian coordinates (using a wider range)
india_df = df[
    (df['Latitude'] >= 5.0) & (df['Latitude'] <= 40.0) &
    (df['Longitude'] >= 65.0) & (df['Longitude'] <= 100.0)
]

print("\nRows after filtering for India:", len(india_df))

if len(india_df) == 0:
    print("\nNo data points found in the specified coordinate range!")
    print("Using all data instead...")
    india_df = df.copy()

# Map categorical columns if needed
weather_map = {
    "Fine no high winds": 1, "Raining no high winds": 2, "Snowing no high winds": 3,
    "Fine + high winds": 4, "Raining + high winds": 5, "Snowing + high winds": 6,
    "Fog or mist": 7, "Fine": 1, "Rain": 2, "Snow": 3, "Mist": 7, "Fog": 7, "Clouds": 1
}
light_map = {
    "Daylight": 1, "Dark - lights lit": 4, "Dark - lights unlit": 5, "Dark - no lighting": 6,
    "daylight": 1, "darkness light lit": 4, "darkness lights unlit": 5, "darkness no lighting": 6
}
road_map = {
    "Dry": 1, "Wet": 2, "Snow": 3, "Frost": 4, "Flood": 5, "Mud": 7,
    "dry": 1, "wet": 2, "snow": 3, "frost": 4, "flood": 5, "mud": 7
}

# Print the first few rows to see the actual column names
print("\nFirst few rows of the data:")
print(df.head())

# Map the categorical columns using the correct column names
india_df['Weather'] = india_df['Weather_Conditions'].map(weather_map).fillna(1)
india_df['Light'] = india_df['Light_Conditions'].map(light_map).fillna(1)
india_df['Road_Surface'] = india_df['Road_Surface_Conditions'].map(road_map).fillna(1)

# Prepare features and target
X = india_df[["Did_Police_Officer_Attend_Scene_of_Accident", "Light", "Road_Surface", "Speed_limit", "Weather", "Latitude", "Longitude"]]
y = india_df["Accident_Severity"]

print("\nShape of features (X):", X.shape)
print("Shape of target (y):", y.shape)

# Encode target if needed
le = LabelEncoder()
y = le.fit_transform(y)

# Fix data imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print("\nAfter resampling:")
print("Shape of features (X_resampled):", X_resampled.shape)
print("Shape of target (y_resampled):", y_resampled.shape)
print("Class distribution after resampling:", pd.Series(y_resampled).value_counts())

# Train and save the model
model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)
model_path = os.path.join(script_dir, "trained_model_india.sav")
joblib.dump(model, model_path)
print(f"\nModel trained and saved as {model_path}") 