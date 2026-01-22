# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, traceback, glob, ast
from itertools import product
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Uni_algo_HP_dict

BASE_FOLDER = "Runs"
# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# print("CUDA available: ", torch.cuda.is_available())
# print("cuDNN version: ", torch.backends.cudnn.version())

def parse_arguments():
    parser = argparse.ArgumentParser(description='HP Tuning')
    
    parser.add_argument('AD_Name', type=str)
    parser.add_argument('-d', '--datasets', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('-f', '--file_list', type=str, default='Datasets/File_List/TSB-AD-U-Tuning-filtered.csv')
    parser.add_argument('-s', '--save', type=str, default=f'{BASE_FOLDER}/HP_tuning')
    
    args = parser.parse_args()
    
    # Validate data
    if not os.path.exists(args.datasets):
        raise ValueError(f"Dataset folder does not exists: {args.datasets}")
    if not os.path.exists(args.file_list):
        raise ValueError(f"File list does not exists: {args.file_list}")
    if args.AD_Name not in Semisupervise_AD_Pool and args.AD_Name not in Unsupervise_AD_Pool:
        raise ValueError(f"No model named {args.AD_Name}")
    
    # Create the save folder
    os.makedirs(args.save, exist_ok=True)
    
    return args

def get_latest_results_file(save_dir, ad_name):
    search_pattern = os.path.join(save_dir, f"HP-{ad_name}-*.csv")
    files = glob.glob(search_pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_processed_combinations(filepath):
    processed = set()
    try:
        df = pd.read_csv(filepath)
        # Ensure necessary columns exist
        if 'file' not in df.columns or 'HP' not in df.columns:
            return processed

        for _, row in df.iterrows():
            try:
                # Convert string representation of dict back to dict
                hp_dict = ast.literal_eval(row['HP'])
                # Create a hashable signature: (filename, frozenset of HP items)
                signature = (row['file'], frozenset(hp_dict.items()))
                processed.add(signature)
            except (ValueError, SyntaxError):
                continue
    except Exception as e:
        print(f"Warning: Could not read existing progress from {filepath}: {e}")
    
    return processed

if __name__ == '__main__':
    args = parse_arguments()

    # Creates the parameters combinations
    Det_HP = Uni_algo_HP_dict[args.AD_Name]
    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    file_list = pd.read_csv(args.file_list)['file_name'].values
    write_csv = []
    
    # Load last run if exists
    processed_sigs = set()
    latest_result_file = get_latest_results_file(args.save, args.AD_Name)
    
    if latest_result_file:
        print(f"Checking existing file for progress: {latest_result_file}")
        processed_sigs = load_processed_combinations(latest_result_file)
        print(f"Found {len(processed_sigs)} completed combinations.")
    
    print(f"Running Hyper-Parameter Tuning for {args.AD_Name}, {len(combinations)} combinations, {len(file_list)} datasets.")
    
    # Loop the files
    start_tuning = time.time()
    try:
        for filename in file_list:
            # Pre-check: If all combinations for this file are done, skip file loading entirely
            all_combs_done = True
            for params in combinations:
                sig = (filename, frozenset(params.items()))
                if sig not in processed_sigs:
                    all_combs_done = False
                    break
            
            if all_combs_done:
                print(f"Skipping {filename} (All HP combinations completed)")
                continue
            
            start_file = time.time()
            
            file_path = os.path.join(args.datasets, filename)
            df = pd.read_csv(file_path).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()

            feats = data.shape[1]
            slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
            train_index = filename.split('.')[0].split('_')[-3]
            data_train = data[:int(train_index), :]
            
            print(f"\tProcessing: {filename}. Training len: {len(data_train)}, Test len: {len(data)}")
            combinations_run_count = 0
            for params in combinations:
                # Check if specific combination is done
                sig = (filename, frozenset(params.items()))
                if sig in processed_sigs:
                    continue
                
                combinations_run_count += 1
                start_params = time.time()
                
                print(f"\n\tUsing: {params}")

                if args.AD_Name in Semisupervise_AD_Pool:
                    output = run_Semisupervise_AD(args.AD_Name, data_train, data, **params)
                elif args.AD_Name in Unsupervise_AD_Pool:
                    output = run_Unsupervise_AD(args.AD_Name, data, **params)
                    
                try:
                    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                    # print('VUS-PR: ', evaluation_result['VUS-PR'])
                    list_w = list(evaluation_result.values())
                except:
                    list_w = [0]*9
                
                list_w.insert(0, params)
                list_w.insert(0, filename)
                write_csv.append(list_w)
                
                print(f"\tVUS-PR: {list_w[4]}, took {(time.time() - start_params):.2f} s")
                            
            if combinations_run_count > 0:
                print(f"\n\tDataset completed in {(time.time() - start_file):.2f} s\n")
        else:
            print(f"\nHP-Tuning completed in {(time.time() - start_tuning):.2f}s.")
    except KeyboardInterrupt:
        print(f"\nManual interrupting the HP-Tuning")
    except Exception as e:
        print(f"\nAn error occoured: {e}")
        print(traceback.format_exc())
        
    if len(write_csv) <= 0:
        print(f"\nNo results to be saved")
    else:
        # Determine save logic
        if latest_result_file:
            save_file_name = latest_result_file
            mode = 'a'
            header = False
            print(f"\nAppending {len(write_csv)} new results to {save_file_name}")
        else:
            save_file_name = f"{args.save}/HP-{args.AD_Name}-{int(time.time())}.csv"
            mode = 'w'
            header = True
            print(f"\nSaving {len(write_csv)} results to {save_file_name}")
        
        df_results = pd.DataFrame(write_csv, columns=['file', 'HP', 'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F'])
        df_results.to_csv(save_file_name, index=False, mode=mode, header=header)

 # Compute and print the best hyperparameters based on average VUS-PR across all datasets
    full_df = pd.read_csv(save_file_name)
    if not full_df.empty and 'HP' in full_df.columns and 'VUS-PR' in full_df.columns:
        try:
            best_hp = full_df.groupby(['HP'])['VUS-PR'].mean().idxmax()
            best_avg_score = full_df.groupby(['HP'])['VUS-PR'].mean().max()
            print(f"\nBest Hyperparameters (average VUS-PR across datasets): {best_hp}")
            print(f"Average VUS-PR: {best_avg_score:.4f}")
        except Exception as e:
            print(f"\nCould not compute best hyperparameters: {e}")
    else:
        print("\nNo results available to compute best hyperparameters.")