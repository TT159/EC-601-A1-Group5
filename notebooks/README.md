# Some open-source projects from Kaggle

Paths and libraries adjusted for [SCC](https://scc-ondemand.bu.edu/)

Run notebooks in the same path as datasets. (/projectnb/ece601/A1-Group5-BrainTumor/dir-brain-tumor/)

Initialize SCC environment (Python3, pytorch, tensorflow) by 

> source initial.sh

and add local Python packages (efficientnet_pytorch, ...) search path by

> source set_python_path.sh

# Some parameters
1. EDA with Animations and Modeling
2. Brain Tumor 3D 
3. Brain Tumor EDA with score
4. Brain Tumor 3D Multimodal CNN - All MRI Type
5. Brain Tumr - Transfer Learning - FLAIR - Kfold
6. Monai - ensemble
EXPLORATORY DATA ANALYSIS (EDA)

## Test Split
1. train_test_split(test_size=0.2)
2. StratifiedKFold(n_split=5,shuffle=True)
3. train_test_split(test_size=0.2)
4. StratifiedKFold(n_split=5,shuffle=True)
5. KFold(n_splits=5,shuffle=True)
6. Exclude 3 sample, train_test_split(test_size=0.2)

## MRI Imaging Sequence
* FLAIR: Fluid Attenuated Inversion Recovery
* T1: longitudinal relaxation time
* T2: transverse relaxation time
* -w: weighted
* -CE: contrast-enhanced
1. FLAIR, T1w, T1wCE
2. FLAIR
3. FLAIR, T1w, T1wCE
4. FLAIR, T1w, T1wCE, T2W
5. FLAIR
6. FLAIR, T1w, T1wCE, T2W

## Data Sampling
1. 10 Uniformly distributed slices across the 3D image
2. 64 centered consecutive image
3. 10 Uniformly distributed slices across the 3D image
4. Middle 24 images
5. Middle 64 images
6. eqaully distributed 64 images

## Image resizing
1. 256x256
2. 256x256
3. 256x256
4. 120x120
5. 240x240
6. 256x256

## Model
1. efficientnet-b0
2. efficientnet in keras
3. efficientnet-b0
4. 4-input CNN model
5. ResNet50 (LSTM)
6. DenseNet121

## Optimizer
1. Adam(lr=0.001)
2. Adam, learning_rate=lr_schedule, lr_schedule=ExponentialDecay(0.0001,decay_rate=0.96)
3. Adadelta(lr=0.001)
4. (default)
5. adam
6. Adam (lr=0.0005), lr_scheduler.EXponentialLR(LR_DECAY=0.9)

## Criterion
1. binary_cross_entropy_with_logits
2. binary_crossentropy
3. binary_cross_entropy_with_logits
4. (default)
5. binary_crossentropy
6. binary_cross_entropy_with_logits

## Accuracy / auc: 
1. accuracy = 0.5263, auc = 0.5291
2. accuracy = 0.5263, auc = 0.5291
3. accuracy = 0.5263, auc - 0.5291
4. accuracy = 0.5, acc = 0.4439
5. accuracy = 0.6184, auc = 0.6565
6. accuracy = 0.5, auc = 0.5796

* case 1,2,3 use almost the same method, so they have the same accuracy and auc.

# Reference:
These are links of open-source projects.
* [🧠Brain Tumor🧠 - EDA with Animations and Modeling](https://www.kaggle.com/ihelon/brain-tumor-eda-with-animations-and-modeling)	
* [🧠Brain Tumor 3D Training](https://www.kaggle.com/ammarnassanalhajali/brain-tumor-3d-training)
* [Brain Tumor EDA with score](https://www.kaggle.com/blade001/brain-tumor-eda-with-score)
* [Brain Tumor 3D Multimodal CNN - All MRI Type](https://www.kaggle.com/michaelfumery/brain-tumor-3d-multimodal-cnn-all-mri-type/)
* [🧠Brain Tumor - Transfert Learning - FLAIR - Kfold](https://www.kaggle.com/michaelfumery/brain-tumor-transfert-learning-flair-kfold)
* [Monai - ensemble](https://www.kaggle.com/mikecho/rsna-miccai-monai-ensemble/notebook)  

Here is our meeting record.
* [Meeting Minutes on Google Doc](https://docs.google.com/document/d/1po9hrZAp04lbLL3IngvLaURMMgkMC6j_7K1OG1VMwKI/edit)
