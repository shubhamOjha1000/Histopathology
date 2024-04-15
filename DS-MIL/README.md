# DSMIL: Dual-stream multiple instance learning networks for tumor detection in Whole Slide Image
Pytorch implementation for the multiple instance learning model described in the paper [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (_CVPR 2021, accepted for oral presentation_).

## Training on your own datasets:-

###  1. Place WSI files as `WSI\[DATASET_NAME]\[CATEGORY_NAME]\SLIDE_NAME.svs`.
>#### Folder structures
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
```
> eg of TCGA Brain dataset having 3 classes (Glioblastoma, Oligodendroglioma, Astrocytoma) :-
```
root
|-- WSI
|   |-- tcga_brain(DATASET_NAME)
|   |   |-- Glioblastoma (CLASS_1)
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- Oligodendroglioma (CLASS_2)
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- Astrocytoma (CLASS_3)
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
```




### 2. Crop patches :-

<img width="614" alt="preprocessing" src="https://github.com/shubhamOjha1000/Histopathology/assets/72977734/c4182364-04e7-4dce-9cb9-c61c97d793c0">

Useful arguments:
```
[-d]    [--dataset]         # Dataset name.

[-p]    [--path_to_WSI]     # Path to WSI folder.

[-e]    [--overlap]         # Amount of overlap between adjacent tiles . By default 0 signifying no overlap.

[-f]    [--format]          # Image format for individual tiles/patch. By default jpeg format.

[-v]    [--slide_format]    # Image format for WSI. By default svs.

[-j]    [--workers]         # The number of worker processes to use for parallel tile generation. This parameter controls the level of parallelism in the tiling process. By default it is set to 4.

[-q]    [--quality]         # JPEG compression quality.This parameter is used when saving image tiles/patches in a compressed format(eg JPEG).Higer values typically result in better quality but leads to larger file size. By default 70.

[-s]    [--tile_size]       # Individula tile/patch size. By default 224.

[-b]    [--base_mag]        # Dataset name.

[-o]    [--objective]       # The objective magnification power of the microscope used to capture the image. If metadata does not present then value is set to 20.

[-t]    [--background_t]    # Threshold for filtering background. Tiles above the threshold are further processed and saved. By default 15.

[-m]    [--magnifications]  # A list of magnification levels for generating image tiles. Each level corresponds to a specific zoom level in zoom pyramid. Suppose a WSI has max resolution 20x, here 0 corresponds to the full/max resolution level. In zoom pyramid each level is created by downscaling the previous level resolution by a factor of 2. So 10x resolution will have level 1 and 5x resolution will have level 2. See fig below of zoom pyramid :-
```
Prepare the patches
```
python deepzoom_tiler.py -d=' Dataset name' -p='Path to WSI folder' -e='Amount of overlap between adjacent tiles' -f='jpeg' -v='svs' -j=4 -q=70 -s=224 -o=20 -t=20 -m 0 2

```




<img width="614" alt="pyramid" src="https://github.com/shubhamOjha1000/Histopathology/assets/72977734/33744a15-67aa-4485-b3db-3be9d6a6b9b3">

Once patch extraction is performed, folder will appear like this :- 
- `pyramid folder` :-
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- pyramid
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_LOW_1
|   |   |   |   |   |   |-- PATCH_HIGH_1.jpeg
|   |   |   |   |   |   |-- ...
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- PATCH_LOW_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...

```


- `single folder` :-
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- single
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...

```







### 3. Compute features using the embedder :-
```
python compute_feats.py --dataset=TCGA-brain --magnification=tree

```
For each bag, there is a .pt file where each row contains the feature of an instance. The .pt is named as "bagID.pt" and put into a folder named "dataset-name/Class_Name/".





### 4. Training :-
```
python train_tcga.py --dataset=TCGA-brain

```






   

