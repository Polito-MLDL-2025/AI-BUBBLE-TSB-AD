# Setup
## Download dataset
```shell
 python ./scripts/download_tsb_ad_datasets.py 
```

# Hyperparameters Fine-tuning

## Chronos2Ada
### Univariate
```shell
python benchmark_exp/HP_Tuning_U.py --dataset_dir ../Datasets/TSB-AD-Datasets/TSB-AD-U \
--file_lsit ../Datasets/File_List/TSB-AD-U-Tuning-s.csv \
--save_dir ../Datasets/TSB-AD-Datasets/hp-tuning-u-s \
--AD_Name Chronos2Ada 
```

### Multivariate

```shell
python benchmark_exp/HP_Tuning_M.py --dataset_dir ../Datasets/TSB-AD-Datasets/TSB-AD-M \
--file_lsit ../Datasets/File_List/TSB-AD-M-Tuning-s.csv \
--save_dir ../Datasets/TSB-AD-Datasets/hp-tuning-m-s \
--AD_Name Chronos2Ada 
```

## Sub_Chronos2Ada
### Univariate
```shell
python benchmark_exp/HP_Tuning_U.py --dataset_dir ../Datasets/TSB-AD-Datasets/TSB-AD-U \
--file_lsit ../Datasets/File_List/TSB-AD-U-Tuning-s.csv \
--save_dir ../Datasets/TSB-AD-Datasets/hp-tuning-u-s \
--AD_Name Sub_Chronos2Ada 
```

### Multivariate

```shell
python benchmark_exp/HP_Tuning_M.py --dataset_dir ../Datasets/TSB-AD-Datasets/TSB-AD-M \
--file_lsit ../Datasets/File_List/TSB-AD-M-Tuning-s.csv \
--save_dir ../Datasets/TSB-AD-Datasets/hp-tuning-m-s \
--AD_Name Sub_Chronos2Ada 
```

---

# Evaluation

## Chronos2Ada
### Univariate
```shell
python benchmark_exp/Run_Detector_U.py --dataset_dir../Datasets/TSB-AD-Datasets/TSB-AD-U \
  --file_lsit ../Datasets/TSB-AD-Datasets/File_List/TSB-AD-U-Eva_test.csv \
  --score_dir ../Datasets/TSB-AD_Datasets/eval_chro2/score/uni \
  --save_dir ../Datasets/TSB-AD-Datasets/eval_chro2/metrics/uni \
  --save False \
  --AD_Name Chronos2Ada
```

### Multivariate

```shell
python benchmark_exp/Run_Detector_M.py --dataset_dir../Datasets/TSB-AD-Datasets/TSB-AD-M \
  --file_lsit ../Datasets/TSB-AD-Datasets/File_List/TSB-AD-M-Eva_test.csv \
  --score_dir ../Datasets/TSB-AD_Datasets/eval_chro2/score/multi \
  --save_dir ../Datasets/TSB-AD-Datasets/eval_chro2/metrics/multi \
  --save False \
  --AD_Name Chronos2Ada
```

## Sub_Chronos2Ada
### Univariate
```shell
python benchmark_exp/Run_Detector_U.py --dataset_dir../Datasets/TSB-AD-Datasets/TSB-AD-U \
  --file_lsit ../Datasets/TSB-AD-Datasets/File_List/TSB-AD-U-Eva_test.csv \
  --score_dir ../Datasets/TSB-AD_Datasets/eval_sub_chro2/score/uni \
  --save_dir ../Datasets/TSB-AD-Datasets/eval_sub_chro2/metrics/uni \
  --save False \
  --AD_Name Sub_Chronos2Ada
```

### Multivariate

```shell
python benchmark_exp/Run_Detector_M.py --dataset_dir../Datasets/TSB-AD-Datasets/TSB-AD-M \
  --file_lsit ../Datasets/TSB-AD-Datasets/File_List/TSB-AD-M-Eva_test.csv \
  --score_dir ../Datasets/TSB-AD_Datasets/eval_sub_chro2/score/multi \
  --save_dir ../Datasets/TSB-AD-Datasets/eval_sub_chro2/metrics/multi \
  --save False \
  --AD_Name Sub_Chronos2Ada
```