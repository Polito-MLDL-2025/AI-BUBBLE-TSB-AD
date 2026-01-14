import pandas as pd

HP_FILE = "Runs/HP_tuning/HP-Chronos_2_AE-1766866397.csv"

def main():
    df = pd.read_csv(HP_FILE)
    best = df.groupby(['HP'])['VUS-PR'].mean().idxmax()
    
    print(f"Best Hyperparameters: {best}")

if __name__ == '__main__':
    main()