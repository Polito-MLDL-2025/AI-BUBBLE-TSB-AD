#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

#%% Constants
MODEL_NAME = "Chronos_2_AE" # 'Chronos_2_AE' or 'Chronos2'
SETTING = "M" # 'U' or 'M'

#%% File paths
TSBAD_BENCHMARK_FILE = f"benchmark_exp/benchmark_eval_results/{'uni' if SETTING == 'U' else 'multi'}_mergedTable_VUS-PR.csv"
CHRONOS2_BENCHMARK_FILE = f"benchmark_exp/benchmark_eval_results/{MODEL_NAME}-{SETTING}.csv"

#%%
def analyze_benchmarks(chronos2_benchmark_file, tsbad_benchmark_file, model_name):
    df_results = pd.read_csv(chronos2_benchmark_file)
    df_results = df_results[['file', 'VUS-PR']].rename(columns={'VUS-PR': model_name})

    df_bench = pd.read_csv(tsbad_benchmark_file)
    meta_cols = ['ts_len', 'anomaly_len', 'num_anomaly', 'avg_anomaly_len', 
                 'anomaly_ratio', 'point_anomaly', 'seq_anomaly']
    df_bench = df_bench.drop(columns=meta_cols, errors='ignore')

    merged_df = pd.merge(df_results, df_bench, on='file', how='inner')
    if merged_df.empty:
        print("No matching datasets found between the two files.")
        return None, None

    merged_df.set_index('file', inplace=True)
    avg_scores = merged_df.mean().sort_values(ascending=False)

    return merged_df, avg_scores

# %% Run analysis
merged_data, avg_data = analyze_benchmarks(CHRONOS2_BENCHMARK_FILE, TSBAD_BENCHMARK_FILE, MODEL_NAME)

display(merged_data)

print("--- Average VUS-PR Ranking (All Datasets) ---")
print(avg_data)

rankings = avg_data.rank(ascending=False)
print(f"\nTarget Model Rank: {int(rankings[MODEL_NAME])} / {len(rankings)}")

# %% Dataset-wise Average VUS-PR Calculation
data_per_type = merged_data[[MODEL_NAME]].reset_index()
data_per_type = data_per_type.sort_values([MODEL_NAME], ascending=[False])
data_per_type['dataset'] = data_per_type['file'].apply(lambda x: x.split('_')[1])
data_per_type = data_per_type.groupby('dataset').mean([MODEL_NAME])
data_per_type = data_per_type.sort_values([MODEL_NAME], ascending=[False])
display(data_per_type)

# %% Plot VUS-PR Score Distribution
if merged_data is not None:
    # Order the columns based on the average score (ranking)
    ordered_columns = avg_data.index
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=merged_data[ordered_columns])
    plt.title('VUS-PR Score Distribution of Models (Ordered by Rank)')
    plt.ylabel('VUS-PR Score across Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No data to plot.")

# %%
