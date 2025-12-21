import pandas as pd
import argparse
import os

def compute_average(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    try:
        df = pd.read_csv(file_path)
        # Select only numeric columns for average calculation
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            print("No numeric columns found in the CSV.")
            return

        averages = numeric_df.mean()
        
        print(f"\nAverage scores for: {file_path}")
        print("-" * 40)
        for metric, value in averages.items():
            print(f"{metric:<20}: {value:.6f}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute average scores from a metrics CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    
    args = parser.parse_args()
    compute_average(args.file_path)

