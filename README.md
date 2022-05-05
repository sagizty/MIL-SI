# MIL-SI
Pancreatic Cancer ROSE Image Classification Based on Multiple Instance Learning with Shuffle Instances

* Results can be view in Archive folder 

* Colab scripts are provided with a sample dataset (MICCAI 2015 chanllange)

# Abstract

The rapid on-site evaluation (ROSE) technique can significantly accelerate the diagnostic workflow of pancreatic cancer by immediately analyzing the fast-stained cytopathological images with on-site pathologists. Computer-aided diagnosis (CAD) using the deep learning method has the potential to solve the problem of insufficient pathology staffing. However, the cancerous patterns of ROSE images vary greatly between different samples, making the CAD task extremely challenging. Besides, due to different staining qualities and various types of acquisition devices, the ROSE images also have complicated perturbations in terms of color distribution, brightness, and contrast. To address these challenges, we proposed a novel multiple instance learning (MIL) approach using shuffle patches as the instances, which adopts the patch-based learning strategy of Vision Transformers. With the shuffle instances of grouped cell patches and their bag-level soft labels, the approach utilizes a MIL head to make the model focus on the features from the pancreatic cancer cells, rather than that from various perturbations in ROSE images. Simultaneously, combined with a classification head, the model can effectively identify the general distributive patterns across different instances. The results demonstrate the significant improvements in the classification accuracy with more accurate attention regions, indicating that the diverse patterns of ROSE images are effectively extracted and the complicated perturbations of ROSE images are greatly eliminated. It also suggests that the MIL with shuffle instances has great potential in the analysis of cytopathological images.


# Method Overview

![MIL-SI](https://user-images.githubusercontent.com/50575108/154795968-9018d2c2-6770-4ddd-9fef-6da0aba9b54e.png)

Overview of our proposed approach MIL-SI, composed of two steps MIL step and CLS step. In the data processing as illustrated in (a), the images will be transformed into patch-es, and the patch annotation label will be calculated based on the corresponding masks. In the MIL step, the bags of patches within a batch will be shuffled while the bags of image patches will remain unchanged in the CLS step. The bags are then composed into images with the soft-label aggregated from the patch-level label. In the 2-step training process in (b), after the feature extraction of the backbone, the patch tokens will be used to regress the bag-level soft-label in the MIL head. In the CLS step, an additional CLS head will be used to predict the categories of the input images based on the class token.

# Results and CAM
![results table](https://user-images.githubusercontent.com/50575108/159670773-6af45d4b-8ed3-43f8-9e0e-1e0f66526657.png)
<img width="1001" alt="CAM results" src="https://user-images.githubusercontent.com/50575108/159670793-d0970b24-70ab-46a4-b683-602cde66c8a6.png">


