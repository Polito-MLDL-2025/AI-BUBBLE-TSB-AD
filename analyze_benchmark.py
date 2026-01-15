import pandas as pd

CHRONOS2AE_FILE = "Runs/Benchmark-Result/Chronos_2_AE-1766874178.csv"
MODEL_NAME = "Chronos-2-AE"
BENCHMARK_FILE = "benchmark_exp/benchmark_eval_results/multi_mergedTable_VUS-PR.csv"

def analyze_benchmarks(results_file=CHRONOS2AE_FILE, benchmark_file=BENCHMARK_FILE):
    global MODEL_NAME
    
    df_results = pd.read_csv(results_file)
    df_results = df_results[['file', 'VUS-PR']].rename(columns={'VUS-PR': MODEL_NAME})

    df_bench = pd.read_csv(benchmark_file)
    meta_cols = ['ts_len', 'anomaly_len', 'num_anomaly', 'avg_anomaly_len', 
                 'anomaly_ratio', 'point_anomaly', 'seq_anomaly']
    df_bench = df_bench.drop(columns=meta_cols, errors='ignore')

    merged_df = pd.merge(df_results, df_bench, on='file', how='inner')
    if merged_df.empty:
        print("No matching datasets found between the two files.")
        return

    merged_df.set_index('file', inplace=True)

    print("--- Average VUS-PR Ranking (All Datasets) ---")
    avg_scores = merged_df.mean().sort_values(ascending=False)
    
    # Add rank
    rankings = avg_scores.rank(ascending=False)
    print(avg_scores)
    print(f"\nTarget Model Rank: {int(rankings[MODEL_NAME])} / {len(rankings)}")

    print("\n--- Rank per Dataset ---")
    ranks_per_file = merged_df.rank(axis=1, ascending=False, method='min')
    target_ranks = ranks_per_file[MODEL_NAME].sort_values()
    
    print(target_ranks)

if __name__ == "__main__":
    analyze_benchmarks()