import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_missing_values(series):
    """Predicts missing values in a series (supports center & end gaps)."""
    # Convert to DataFrame for easier handling
    df = pd.DataFrame({'value': series})
    df['index'] = np.arange(len(df))  # Position-based prediction
    
    # Identify missing values
    missing_mask = df['value'].isna()
    
    if not missing_mask.any():
        print("✅ No missing values found!")
        return series
    
    # Split into known and unknown values
    known = df[~missing_mask]
    unknown = df[missing_mask]
    
    # Train model on known data
    model = LinearRegression()
    model.fit(known[['index']], known['value'])
    
    # Predict missing values
    predicted = model.predict(unknown[['index']])
    df.loc[missing_mask, 'value'] = predicted
    
    print("\n🔍 Predicted Missing Values:")
    for idx, val in zip(unknown.index, np.round(predicted, 2)):
        print(f"Position {idx + 1}: {val}")
    
    return df['value'].tolist()

# Example usage
if __name__ == "__main__":
    print("\n🔧 MISSING VALUE PREDICTOR TOOL 🔧")
    print("----------------------------------")
    print("Enter your series with 'nan' for missing values (e.g., '10, 20, nan, 40')")
    
    while True:
        user_input = input("\nEnter series (comma-separated): ").strip()
        try:
            # Parse input (supports "10, 20, nan, 40" or "[10, 20, nan, 40]")
            series = [
                float(x) if x.strip().lower() not in ['nan', 'none', ''] else np.nan
                for x in user_input.replace('[','').replace(']','').split(',')
            ]
            
            # Predict missing values
            completed_series = predict_missing_values(series)
            
            # Print final result
            print("\n🎯 Completed Series:")
            print(np.round(completed_series, 2))
            
            # Ask to continue
            another = input("\nPredict another series? (y/n): ").lower().strip()
            if another != 'y':
                print("\nThanks for using the tool! 👋")
                break
                
        except ValueError:
            print("❌ Invalid input! Example: '10, 20, nan, 40'")