# MIL-SI
Pancreatic Cancer ROSE Image Classification Based on Multiple Instance Learning with Shuffle Instances

* Results can be view in Archive folder 

* Colab scripts are provided with a sample dataset (MICCAI 2015 chanllange)

# Abstract

The rapid on-site evaluation (ROSE) technique can significantly accelerate the diagnostic workflow of pancreatic cancer by immediately analyzing the fast-stained cytopathological images with on-site pathologists. Computer-aided diagnosis (CAD) using the deep learning method has the potential to solve the problem of insufficient pathology staffing. However, the cancerous patterns of ROSE images vary greatly between different samples, making the CAD task extremely challenging. Besides, due to different staining qualities and various types of acquisition devices, the ROSE images also have complicated perturbations in terms of color distribution, brightness, and contrast. To address these challenges, we proposed a novel multiple instance learning (MIL) approach using shuffle patches as the instances, which adopts the patch-based learning strategy of Vision Transformers. With the shuffle instances of grouped cell patches and their bag-level soft labels, the approach utilizes a MIL head to make the model focus on the features from the pancreatic cancer cells, rather than that from various perturbations in ROSE images. Simultaneously, combined with a classification head, the model can effectively identify the general distributive patterns across different instances. The results demonstrate the significant improvements in the classification accuracy with more accurate attention regions, indicating that the diverse patterns of ROSE images are effectively extracted and the complicated perturbations of ROSE images are greatly eliminated. It also suggests that the MIL with shuffle instances has great potential in the analysis of cytopathological images.


# Method Overview

![MIL-SI](https://user-images.githubusercontent.com/50575108/154795968-9018d2c2-6770-4ddd-9fef-6da0aba9b54e.png)

Overview of our proposed approach MIL-SI, composed of two steps MIL step and CLS step. In the data processing as illustrated in (a), the images will be transformed into patch-es, and the patch annotation label will be calculated based on the corresponding masks. In the MIL step, the bags of patches within a batch will be shuffled while the bags of image patches will remain unchanged in the CLS step. The bags are then composed into images with the soft-label aggregated from the patch-level label. In the 2-step training process in (b), after the feature extraction of the backbone, the patch tokens will be used to regress the bag-level soft-label in the MIL head. In the CLS step, an additional CLS head will be used to predict the categories of the input images based on the class token.


# Results on the test set

## MIL-SI

| Model                               | Model info | MIL Info        | size     | Acc (%) | Precision (%) | Recall (%) | Sensitivity (%) | Specificity (%) | NPV (%) | F1_score (%) |
| ----------------------------------- | ---------- | --------------- | -------- | ------- | ------------- | ---------- | --------------- | --------------- | ------- | ------------ |
| ViT_384_401_PT_lf05_b4_p32_ROSE_MIL | MIL ViT    | CLS+CLS_MIL+MIL | 384, P32 | 94.00   | 91.98         | 90.68      | 90.68           | 95.77           | 95.05   | 91.32        |



## SOTA models

| Model                                       | Model info       | MIL Info | size | Acc (%) | Precision (%) | Recall (%) | Sensitivity (%) | Specificity (%) | NPV (%) | F1_score (%) |
| ------------------------------------------- | ---------------- | -------- | ---- | ------- | ------------- | ---------- | --------------- | --------------- | ------- | ------------ |
| vgg16_384_401_PT_lf05_b4_ROSE_CLS           | VGG 16           | CLS      | 384  | 90.65   | 86.27         | 87.01      | 87.01           | 92.60           | 93.02   | 86.64        |
| vgg19_384_401_PT_lf05_b4_ROSE_CLS           | VGG 19           | CLS      | 384  | 90.06   | 90.42         | 79.94      | 79.94           | 95.47           | 89.90   | 84.86        |
| mobilenetv3_384_401_PT_lf05_b4_ROSE_CLS     | Mobilenet v3     | CLS      | 384  | 89.57   | 91.06         | 77.68      | 77.68           | 95.92           | 88.94   | 83.84        |
| efficientnet_b3_384_401_PT_lf05_b4_ROSE_CLS | Efficientnet_b3  | CLS      | 384  | 89.57   | 85.03         | 85.03      | 85.03           | 91.99           | 91.99   | 85.03        |
| ResNet50_384_401_PT_lf05_b4_ROSE_CLS        | ResNet50         | CLS      | 384  | 90.75   | 87.36         | 85.88      | 85.88           | 93.35           | 92.51   | 86.61        |
| inceptionv3_384_401_PT_lf05_b4_ROSE_CLS     | Inception v3     | CLS      | 384  | 90.75   | 86.72         | 86.72      | 86.72           | 92.90           | 92.90   | 86.72        |
| xception_384_401_PT_lf05_b4_ROSE_CLS        | Xception         | CLS      | 384  | 90.94   | 91.46         | 81.64      | 81.64           | 95.92           | 90.71   | 86.27        |
| swin_b_384_401_PT_lf05_b4_ROSE_CLS          | Swin Transformer | CLS      | 384  | 89.17   | 86.75         | 81.36      | 81.36           | 93.35           | 90.35   | 83.97        |
| ViT_384_401_PT_lf05_b4_ROSE_CLS             | ViT              | CLS      | 384  | 90.65   | 88.20         | 84.46      | 84.46           | 93.96           | 91.88   | 86.29        |
| conformer_384_401_PT_lf05_b4_ROSE_CLS       | Conformer        | CLS      | 384  | 89.67   | 90.82         | 78.25      | 78.25           | 95.77           | 89.17   | 84.07        |
| cross_former_224_401_PT_lf05_b4_ROSE_CLS    | Cross_former     | CLS      | 384  | 89.67   | 86.94         | 82.77      | 82.77           | 93.35           | 91.02   | 84.80        |
| PC_Hybrid2_384_401_PT_lf05_b4_ROSE_CLS      | MSHT             | CLS      | 384  | 90.65   | 90.60         | 81.64      | 81.64           | 95.47           | 90.67   | 85.88        |



## Counterpart augmentations

| Model                                  | Model info | MIL Info | size | Acc (%) | Precision (%) | Recall (%) | Sensitivity (%) | Specificity (%) | NPV (%) | F1_score (%) |
| -------------------------------------- | ---------- | -------- | ---- | ------- | ------------- | ---------- | --------------- | --------------- | ------- | ------------ |
| ViT_384_401_PT_lf05_b4_ROSE_CutMix_CLS | ViT        | CLS      | 384  | 92.72   | 89.55         | 89.55      | 89.55           | 94.41           | 94.41   | 89.55        |
| ViT_384_401_PT_lf05_b4_ROSE_Cutout_CLS | ViT        | CLS      | 384  | 92.32   | 91.07         | 86.44      | 86.44           | 95.47           | 92.94   | 88.70        |
| ViT_384_401_PT_lf05_b4_ROSE_Mixup_CLS  | ViT        | CLS      | 384  | 92.52   | 88.83         | 89.83      | 89.83           | 93.96           | 94.53   | 89.33        |



## Different head structure

| Model                                   | Model info               | MIL Info                            | size     | Acc (%) | Precision (%) | Recall (%) | Sensitivity (%) | Specificity (%) | NPV (%) | F1_score (%) |
| --------------------------------------- | ------------------------ | ----------------------------------- | -------- | ------- | ------------- | ---------- | --------------- | --------------- | ------- | ------------ |
| PC_ViT_384_401_PT_lf05_b4_ROSE_CLS      | ViT                      | CLS                                 | 384      | 90.65   | 88.20         | 84.46      | 84.46           | 93.96           | 91.88   | 86.29        |
| ViT_384_401_PT_lf05_b4_p32_NS_ROSE_MIL  | MIL ViT (no shuffle MIL) | CLS+CLS_MIL                         | 384, P32 | 92.13   | 90.06         | 87.01      | 87.01           | 94.86           | 93.18   | 88.51        |
| ViT_384_401_PT_lf05_b4_p32_NCLSMIL_ROSE | MIL ViT                  | CLS+MIL, no cls step MIL regression | 384, P32 | 93.41   | 91.59         | 89.27      | 89.27           | 95.62           | 94.34   | 90.41        |
| ViT_384_401_PT_lf05_b4_p32_ROSE_MIL     | MIL ViT                  | CLS+CLS_MIL+MIL                     | 384, P32 | 94.00   | 91.98         | 90.68      | 90.68           | 95.77           | 95.05   | 91.32        |



## Different patch size

| Model                                | Model info | MIL Info        | size      | Acc (%) | Precision (%) | Recall (%) | Sensitivity (%) | Specificity (%) | NPV (%) | F1_score (%) |
| ------------------------------------ | ---------- | --------------- | --------- | ------- | ------------- | ---------- | --------------- | --------------- | ------- | ------------ |
| ViT_384_401_PT_lf05_b4_p16_ROSE_MIL  | MIL ViT    | CLS+CLS_MIL+MIL | 384, P16  | 93.60   | 92.88         | 88.42      | 88.42           | 96.37           | 93.96   | 90.59        |
| ViT_384_401_PT_lf05_b4_p32_ROSE_MIL  | MIL ViT    | CLS+CLS_MIL+MIL | 384, P32  | 94.00   | 91.98         | 90.68      | 90.68           | 95.77           | 95.05   | 91.32        |
| ViT_384_401_PT_lf05_b4_p64_ROSE_MIL  | MIL ViT    | CLS+CLS_MIL+MIL | 384, P64  | 93.11   | 91.04         | 88.98      | 88.98           | 95.32           | 94.18   | 90.00        |
| ViT_384_401_PT_lf05_b4_p128_ROSE_MIL | MIL ViT    | CLS+CLS_MIL+MIL | 384, P128 | 92.62   | 92.92         | 85.31      | 85.31           | 96.53           | 92.47   | 88.95        |



## Different head balance

| Model                                  | Model info | MIL Info              | size     | Acc (%) | Precision (%) | Recall (%) | Sensitivity (%) | Specificity (%) | NPV (%) | F1_score (%) |
| -------------------------------------- | ---------- | --------------------- | -------- | ------- | ------------- | ---------- | --------------- | --------------- | ------- | ------------ |
| ViT_384_401_PT_lf05_b4_p32_MIL_05_ROSE | MIL ViT    | CLS+0.5CLS_MIL+0.5MIL | 384, P32 | 91.73   | 84.62         | 93.22      | 93.22           | 90.94           | 96.17   | 88.71        |
| ViT_384_401_PT_lf05_b4_p32_MIL_12_ROSE | MIL ViT    | CLS+1.2CLS_MIL+1.2MIL | 384, P32 | 92.52   | 87.77         | 91.24      | 91.24           | 93.20           | 95.22   | 89.47        |
| ViT_384_401_PT_lf05_b4_p32_MIL_15_ROSE | MIL ViT    | CLS+1.5CLS_MIL+1.5MIL | 384, P32 | 93.31   | 91.81         | 88.70      | 88.70           | 95.77           | 94.07   | 90.23        |
| ViT_384_401_PT_lf05_b4_p32_MIL_18_ROSE | MIL ViT    | CLS+1.8CLS_MIL+1.8MIL | 384, P32 | 93.41   | 91.12         | 89.83      | 89.83           | 95.32           | 94.60   | 90.47        |
| ViT_384_401_PT_lf05_b4_p32_MIL_25_ROSE | MIL ViT    | CLS+2.5CLS_MIL+2.5MIL | 384, P32 | 93.50   | 92.35         | 88.70      | 88.70           | 96.07           | 94.08   | 90.49        |
| ViT_384_401_PT_lf05_b4_p32_MIL_30_ROSE | MIL ViT    | CLS+3.0CLS_MIL+3.0MIL | 384, P32 | 93.60   | 91.40         | 90.11      | 90.11           | 95.47           | 94.75   | 90.75        |

# Attention visuallization by grad-CAM

<img width="1001" alt="CAM results" src="https://user-images.githubusercontent.com/50575108/159670793-d0970b24-70ab-46a4-b683-602cde66c8a6.png">

# CAM on shuffled instances
<img width="607" alt="CAM on shuffled instances" src="https://user-images.githubusercontent.com/50575108/183827312-9f25284a-06da-4cd9-959b-1aac6d6c787e.png">

# bad caces
<img width="593" alt="bad caces" src="https://user-images.githubusercontent.com/50575108/183827366-6207da21-2073-4763-b853-07e679b31271.png">

# CAM of different patch settings
<img width="922" alt="CAM of different patch settings" src="https://user-images.githubusercontent.com/50575108/183827405-d4d7260a-02ef-4da8-b0dc-cb3bc8bbb2f8.png">

# CAM of different settings on shuffled samples
<img width="631" alt="CAM of different settings on shuffled samples" src="https://user-images.githubusercontent.com/50575108/183827453-9d2f2e50-bb3e-4a10-bea5-74b0f3acb761.png">

More samples can be viewed in the folder of Archive


