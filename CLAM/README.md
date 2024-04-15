# CLAM

Pytorch implementation for the multiple instance learning model described in the paper :- [Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images.](https://arxiv.org/abs/2004.09666) Nature Biomedical Engineering 

## Download TCGA-Brain dataset :- 
```
path_to_gdc-client/gdc-client download -m path_to_gdc_manifest_brain_file_.txt

```
You can either download TCGA-Brain from the website or take it from this repo :- 
```
gdc_manifest_brain_full.txt
```
Get lables for TCGA-Brain from the website :- 
```
python get_labels.py

```
### 1. Place WSI files as :- 
```
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```


### 2. Create Patches :- 
```
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 

```

### 3. Compute features using the embedder :-
```
python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs

```

### 4. Training :- 
```
python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_2_tumor_subtyping_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --log_data --subtyping --data_root_dir DATA_ROOT_DIR --embed_dim 1024

```





### Results on TCGA-Brain for three way classification :- 

| Metric       | Accuracy    | AUC    |
|--------------|-------------|--------|
| **Values**  | 94.7%      | 94.8 |



