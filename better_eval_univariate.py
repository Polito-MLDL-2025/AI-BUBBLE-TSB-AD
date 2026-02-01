# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging, traceback, glob
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

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
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    
    parser.add_argument('AD_Name', type=str)
    parser.add_argument('-d', '--datasets', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('-f', '--file_list', type=str, default='Datasets/File_List/TSB-AD-U-Eva-filtered.csv')
    parser.add_argument('-S', '--score', type=str, default=f'{BASE_FOLDER}/Scores/Univariate')
    parser.add_argument('-s', '--save', type=str, default=f'{BASE_FOLDER}/Benchmark-Result')
    
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
    os.makedirs(args.score, exist_ok=True)
    
    return args

def get_latest_results_file(save_dir, ad_name):
    search_pattern = os.path.join(save_dir, f"{ad_name}-*.csv")
    files = glob.glob(search_pattern)
    if not files:
        return None
    # Sort by modification time, newest first
    return max(files, key=os.path.getmtime)

if __name__ == '__main__':
    args = parse_arguments()
    
    score_folder = os.path.join(args.score, args.AD_Name)
    log_file = f'{score_folder}/000_run_{args.AD_Name}.log'
    
    os.makedirs(score_folder, exist_ok = True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    write_csv = []
    
    # Load last run if exists
    processed_files = set()
    latest_result_file = get_latest_results_file(args.save, args.AD_Name)
    if latest_result_file:
        try:
            df_existing = pd.read_csv(latest_result_file)
            if 'file' in df_existing.columns:
                processed_files = set(df_existing['file'].values)
                print(f"Resuming from {latest_result_file}. Found {len(processed_files)} processed files.")
            else:
                print(f"Warning: Existing file {latest_result_file} does not have 'file' column. Starting fresh.")
        except Exception as e:
            print(f"Error reading existing result file: {e}. Starting fresh.")
            latest_result_file = None
    
    print(f"Running Benchmark for {args.AD_Name}, hyper-parameters {Optimal_Det_HP}, {len(file_list)} datasets.")
    
    start_benchmark = time.time()
    try:
        for filename in file_list:
            # Skip if already processed
            if filename in processed_files: continue
            
            start_file = time.time()
            
            score_file = f"{score_folder}/{filename.split('.')[0]}.npy"
            file_path = os.path.join(args.datasets, filename)
            df = pd.read_csv(file_path).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()

            feats = data.shape[1]
            slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
            train_index = filename.split('.')[0].split('_')[-3]
            data_train = data[:int(train_index), :]

            if os.path.exists(score_file):
                print(f"\nFound score file at {score_file}.")
                output = np.load(score_file)
            else:
                print(f"\nProcessing: {filename}. Training len: {len(data_train)}, Test len: {len(data)}\n")
                if args.AD_Name in Semisupervise_AD_Pool:
                    output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
                elif args.AD_Name in Unsupervise_AD_Pool:
                    output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
            
            
            if isinstance(output, np.ndarray):
                logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {time.time() - start_file}s at length {len(label)}')
                np.save(score_file, output)
            else:
                logging.error(f'At {filename}: ' + output)

            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                # print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except:
                list_w = [0]*9
            
            run_time = time.time() - start_file
            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)
            
            print(f"\t{filename} -> VUS-PR: {list_w[4]}, took {(run_time):.2f} s")
        else:
            print(f"\nBenchmark completed in {(time.time() - start_benchmark):.2f}s.")
    except KeyboardInterrupt:
        print(f"\nManual interrupting the benchmark")
    except Exception as e:
        print(f"\nAn error occoured: {e}")
        print(traceback.format_exc())
    
    if len(write_csv) <= 0:
        print(f"\nNo results to be saved")
    else:
        if latest_result_file:
            save_file_name = latest_result_file
            mode = 'a'
            header = False
            print(f"\nAppending {len(write_csv)} results to existing file: {save_file_name}")
        else:
            save_file_name = f"{args.save}/{args.AD_Name}-{int(time.time())}.csv"
            mode = 'w'
            header = True
            print(f"\nSaving {len(write_csv)} results to new file: {save_file_name}")
        
        df_results = pd.DataFrame(write_csv, columns=['file', 'Time', 'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC', 'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F'])
        df_results.to_csv(save_file_name, index=False, mode=mode, header=header)