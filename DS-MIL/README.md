# DSMIL: Dual-stream multiple instance learning networks for tumor detection in Whole Slide Image
Pytorch implementation for the multiple instance learning model described in the paper [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (_CVPR 2021, accepted for oral presentation_).

## Training on your own datasets:-

1. Place WSI files as `WSI\[DATASET_NAME]\[CATEGORY_NAME]\SLIDE_NAME.svs`.
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

2. Crop patches :-

<img width="614" alt="preprocessing" src="https://github.com/shubhamOjha1000/Histopathology/assets/72977734/c4182364-04e7-4dce-9cb9-c61c97d793c0">

<img width="614" alt="pyramid" src="https://github.com/shubhamOjha1000/Histopathology/assets/72977734/33744a15-67aa-4485-b3db-3be9d6a6b9b3">


3. Train an embedder :- 

4. Compute features using the embedder :- 

5. Training :- 