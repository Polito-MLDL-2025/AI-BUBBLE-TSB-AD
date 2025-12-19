import sys
import os
import pandas as pd
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add the current directory to python path to make sure TSB_AD is importable
sys.path.append(os.getcwd())

from TSB_AD.model_wrapper import run_Chronos_2_AE

def main():
    print("Starting Chronos 2 AE Check Script...")
    
    # Load sample data
    data_path = 'Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        data = df['Data'].values.reshape(-1, 1).astype(float)
        
        # Load Labels if they exist, otherwise create dummy zeros
        if 'Label' in df.columns:
            labels = df['Label'].values.astype(int)
        else:
            print("Warning: 'Label' column not found. Assuming all normal (0).")
            labels = np.zeros(len(data))
            
        print(f"Data loaded. Shape: {data.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Split into train/test following the filename format
    split_idx = int(data_path.split('_')[-3])
    
    # Data for training phase
    data_train = data[:split_idx]
    
    # Data for testing == Full dataset
    data_test = data    
    labels_test = labels # User for plotting
    
    print(f"Train shape: {data_train.shape}")
    print(f"Test shape: {data_test.shape}")

    # Call the model wrapper
    print("\nAttempting to run run_Chronos_2_AE (Semi-Supervised)...")
    try:
        scores = run_Chronos_2_AE(
            data_train=data_train,
            data_test=data_test,
            slidingWindow=100,
            head_type='vae',
            latent_dim=32,
        )
        
        print("\nExecution finished successfully!")
        
        if scores is not None:
            print(f"Result scores shape: {scores.shape}")
            
            print("Generating plot...")
            
            # Create a figure with 2 subplots sharing the X-axis
            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Subplot 1: Original Time Series (Test Set) with Anomalies
            axs[0].plot(data_test, label='Time Series', color='blue', alpha=0.6, linewidth=1)
            anomaly_indices = np.where(labels_test == 1)[0]
            if len(anomaly_indices) > 0:
                axs[0].scatter(anomaly_indices, data_test[anomaly_indices], color='red', s=20, label='Ground Truth Anomaly', zorder=5)
            
            axs[0].set_title("Test Data & Ground Truth Anomalies")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # Subplot 2: Anomaly Scores
            axs[1].plot(scores, label='Reconstruction Error (Score)', color='orange', linewidth=1)
            axs[1].set_title("Anomaly Scores (Chronos-2 AE)")
            axs[1].set_xlabel("Time Step")
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("check_vis_results.png")
            print("Plot saved to 'check_vis_results.png'")
            # plt.show()
        else:
            print("Result is None.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()