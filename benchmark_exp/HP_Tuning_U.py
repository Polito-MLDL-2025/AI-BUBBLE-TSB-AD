# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os
import itertools
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Uni_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

METRIC_COLUMNS = [
    'AUC-PR',
    'AUC-ROC',
    'VUS-PR',
    'VUS-ROC',
    'Standard-F1',
    'PA-F1',
    'Event-based-F1',
    'R-based-F1',
    'Affiliation-F',
]

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='HP Tuning')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='../Datasets/File_List/TSB-AD-U-Tuning.csv')
    parser.add_argument('--save_dir', type=str, default='eval/HP_tuning/uni/')
    parser.add_argument('--AD_Name', type=str, default='IForest')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    output_csv = os.path.join(args.save_dir, f'{args.AD_Name}.csv')

    processed_pairs = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv, usecols=['file', 'HP'])
            processed_pairs = set(
                zip(existing_df['file'].astype(str), existing_df['HP'].astype(str))
            )
            print(f'Loaded {len(processed_pairs)} existing results from {output_csv}')
        except Exception as exc:
            print(f'Warning: failed to read existing results from {output_csv}: {exc}')

    file_list = pd.read_csv(args.file_lsit)['file_name'].values

    Det_HP = Uni_algo_HP_dict[args.AD_Name]

    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for filename in file_list:
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        # print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        for params in combinations:
            hp_str = str(params)
            if (filename, hp_str) in processed_pairs:
                print(f'Skipping {filename} HP:{hp_str} as it has already been processed')
                continue

            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **params)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **params)
            else:
                raise Exception(f"{args.AD_Name} is not defined")
                
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
            except Exception:
                evaluation_result = {}

            metrics_row = {k: 0 for k in METRIC_COLUMNS}
            metrics_row.update(evaluation_result)
            row = {'file': filename, 'HP': hp_str, **metrics_row}

            pd.DataFrame([row], columns=['file', 'HP'] + METRIC_COLUMNS).to_csv(
                output_csv,
                mode='a',
                header=not os.path.exists(output_csv),
                index=False,
            )
            processed_pairs.add((filename, hp_str))
