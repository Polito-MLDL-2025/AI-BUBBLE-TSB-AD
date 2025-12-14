
import sys
import os
import pandas as pd
import numpy as np
import traceback

# Add the current directory to python path to make sure TSB_AD is importable
sys.path.append(os.getcwd())

from TSB_AD.model_wrapper import run_Chronos_2_AE

def main():
    print("Starting Chronos 2 AE Check Script...")
    
    # Load sample data
    # Using one of the files found in the file structure
    data_path = 'Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        data = df['Data'].values.reshape(-1, 1).astype(float)
        print(f"Data loaded. Shape: {data.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Split into train/test
    # Simple split for testing purposes
    split_idx = len(data) // 2
    data_train = data[:split_idx]
    data_test = data[split_idx:]
    
    print(f"Train shape: {data_train.shape}")
    print(f"Test shape: {data_test.shape}")

    # Call the model wrapper
    print("\nAttempting to run run_Chronos_2_AE...")
    try:
        scores = run_Chronos_2_AE(
            data_train=data_train,
            data_test=data_test,
            slidingWindow=100,
            head_type='ae',
            latent_dim=32,
            epochs=1  # Reduced epochs for quick check
        )
        
        print("\nExecution finished successfully!")
        if scores is not None:
             print(f"Result scores shape: {scores.shape}")
             print(f"First 10 scores: {scores[:10]}")
        else:
             print("Result is None.")

    except Exception as e:
        print("\nExecution FAILED.")
        print("Error details:")
        traceback.print_exc()

    # TODO: Implement effective windowing + stride to produce anomaly distribution

if __name__ == "__main__":
    main()
