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


