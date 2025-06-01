import pandas as pd
import numpy as np
import random
import os

def add_location_data(input_file, output_file):
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths
    input_path = os.path.join(current_dir, input_file)
    output_path = os.path.join(current_dir, output_file)
    
    # Read the existing dataset
    print(f"Reading input file from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Generate realistic UK latitude and longitude ranges
    # UK coordinates range approximately:
    # Latitude: 49.9 to 60.9
    # Longitude: -8.6 to 1.8
    print("Generating location data...")
    df['Latitude'] = np.random.uniform(49.9, 60.9, size=len(df))
    df['Longitude'] = np.random.uniform(-8.6, 1.8, size=len(df))
    
    # Add some correlation between location and accident severity
    # Higher severity accidents are more likely in urban areas (southern UK)
    severity_weights = {
        'Fatal': 0.6,    # More likely in southern UK
        'Serious': 0.3,  # Medium likelihood
        'Slight': 0.1    # Less likely in southern UK
    }
    
    # Adjust latitudes based on severity
    for severity, weight in severity_weights.items():
        mask = df['Accident_Severity'] == severity
        df.loc[mask, 'Latitude'] = df.loc[mask, 'Latitude'] * (1 - weight) + 49.9 * weight
    
    # Save the updated dataset
    print(f"Saving updated dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Updated dataset saved to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print("\nAccident Severity Distribution:")
    print(df['Accident_Severity'].value_counts())
    print("\nLocation Statistics:")
    print(f"Latitude range: {df['Latitude'].min():.2f} to {df['Latitude'].max():.2f}")
    print(f"Longitude range: {df['Longitude'].min():.2f} to {df['Longitude'].max():.2f}")

if __name__ == "__main__":
    input_file = "cleaned_accident_data_preprocessed (4).csv"
    output_file = "cleaned_accident_data_with_location.csv"
    add_location_data(input_file, output_file) 